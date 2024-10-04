import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeKernelLoss(nn.Module):
    def __init__(self, tau_s=5.0):
        super(SpikeKernelLoss, self).__init__()
        self.tau_s = tau_s

    def forward(self, outputs, target):
        target_psp = self._psp(target, self.tau_s)
        delta = outputs - target_psp
        return 0.5 * torch.sum(delta**2)

    def _psp(self, inputs, tau_s):
        n_steps = inputs.shape[-1]
        shape = inputs.shape
        syn = torch.zeros(shape[:-1], dtype=inputs.dtype, device=inputs.device)
        syns = torch.zeros(*shape, dtype=inputs.dtype, device=inputs.device)

        for t in range(n_steps):
            syn = syn - syn / tau_s + inputs[..., t]
            syns[..., t] = syn / tau_s

        return syns

class SpikeCountLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, target, desired_count, undesired_count):
        n_steps = outputs.shape[4]
        out_count = torch.sum(outputs, dim=4)

        delta = (out_count - target) / n_steps

        # # undesired_count에 대한 mask 적용
        # mask = torch.ones_like(out_count)
        # mask[target == undesired_count] = 0
        # mask[delta < 0] = 0
        # delta[mask == 1] = 0

        # mask = torch.ones_like(out_count)
        # mask[target == desired_count] = 0
        # mask[delta > 0] = 0
        # delta[mask == 1] = 0

        delta = delta.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class SpikeCountLoss(nn.Module):
    def __init__(self, desired_count, undesired_count, focal = False, gamma = 2.0):
        super(SpikeCountLoss, self).__init__()
        self.desired_count = desired_count
        self.undesired_count = undesired_count
        self.focal = focal
        self.gamma = gamma
        
    def forward(self, outputs, labels):
        
        target = torch.ones_like(outputs[...,0])*self.undesired_count
        
        target.scatter_(1, 
                        labels.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                        torch.full_like(target, self.desired_count))
        
        delta = SpikeCountLossFunction.apply(outputs, target, self.desired_count, self.undesired_count)
        
        if self.focal:
            pt = torch.exp(-0.5 * torch.sum(delta ** 2, dim=1))  # 각 샘플에 대한 예측 확률 추정
            focal_mse_loss = ((1 - pt) ** self.gamma) * 0.5 * torch.sum(delta ** 2)


        
        return focal_mse_loss.mean()


class DynamicSpikeCountLoss(nn.Module):
    def __init__(self, desired_count, undesired_count):
        super(DynamicSpikeCountLoss, self).__init__()
        self.desired_count = desired_count
        self.undesired_count = undesired_count

    def forward(self, outputs, labels):
        # 각 배치 내에서 클래스별 데이터 빈도 계산
        unique_labels, label_counts = labels.unique(return_counts=True)
        total_labels = labels.size(0)

        # 각 클래스에 대한 가중치 계산
        weights = (total_labels - label_counts.float()) / total_labels
        
        # target 계산
        target = torch.ones_like(outputs[...,0]) * self.undesired_count
        target.scatter_(1, 
                        labels.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                        torch.full_like(target, self.desired_count))

        # 가중치 적용
        weight_map = torch.zeros_like(outputs[..., 0])
        for i, label in enumerate(unique_labels):
            weight_map[labels == label] = weights[i]

        # SpikeCountLossFunction에서 가중치 적용
        delta = SpikeCountLossFunction.apply(outputs, target, self.desired_count, self.undesired_count, weight_map)
        return 1 / 2 * torch.sum(delta**2)

class SpikeSoftmaxLoss(nn.Module):
    def __init__(self):
        super(SpikeSoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        delta = F.log_softmax(outputs.sum(dim=4).squeeze(-1).squeeze(-1), dim=1)
        return self.criterion(delta, target)

import torch
import torch.nn as nn
import torch.nn.functional as F

from eva.tsslbp import TSSLBP


class SNNConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, use_tsslbp=True, tau_m=5, 
                 tau_s=1, threshold=1.0, n_steps =5):
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel_size = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                "kernelSize can only be of 1 or 2 dimension. It was: {}".format(
                    kernel_size.shape
                )
            )

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                "stride can be either int or tuple of size 2. It was: {}".format(
                    stride.shape
                )
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                "padding can be either int or tuple of size 2. It was: {}".format(
                    padding.shape
                )
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                "dilation can be either int or tuple of size 2. It was: {}".format(
                    dilation.shape
                )
            )

        super(SNNConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False  # Conv3d 생성자에 bias=False 명시
        )
        
        self.use_tsslbp = use_tsslbp
        # TSSLBP 설정 초기화
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = threshold
        self.n_steps = n_steps
        
        # syn_a 초기화
        self.register_buffer('syn_a', self.init_syn_a(n_steps, tau_s))
        self.register_buffer('membrain_input', None)
        
    def extra_repr(self):
        org_repr = super(SNNConv3d, self).extra_repr()
        
        return f"{org_repr}, use_tsslbp={self.use_tsslbp}, tau_m={self.tau_m}, tau_s={self.tau_s}, threshold={self.threshold}, n_steps={self.n_steps}"
    
    def init_syn_a(self, n_steps, tau_s):
        syn_a = torch.zeros((1, 1, 1, 1, n_steps), dtype=torch.float32)
        syn_a[..., 0] = 1
        for t in range(n_steps - 1):
            syn_a[..., t + 1] = (
                syn_a[..., t] - syn_a[..., t] / tau_s
            )
        syn_a /= tau_s
        return syn_a

    def forward(self, x):
        x = F.conv3d(
            x,
            self.weight,
            None,  # bias는 사용되지 않으므로 None으로 명시
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        
        self.membrain_input = x
        
        # x와 동일한 device와 dtype으로 syn_a 복사
        device_syn_a = self.syn_a.to(x.device).type(x.dtype)
        if self.use_tsslbp:
            # TSSLBP 설정이 제대로 설정되었는지 확인
            x = TSSLBP.apply(x, self.tau_m, self.tau_s, self.threshold, device_syn_a)
        return x

    def weight_clipper(self, clip_value=4):
        with torch.no_grad():
            self.weight.clamp_(-clip_value, clip_value)
            
            
class SNNLinear(nn.Linear):
    def __init__(self, in_features, out_features, use_tsslbp=True, tau_m=5, 
                 tau_s=1, threshold=1.0, n_steps = 5):

        super(SNNLinear, self).__init__(
            in_features, out_features, bias=False
        )

        # TSSLBP 관련 설정
        self.use_tsslbp = use_tsslbp
        # TSSLBP 설정 초기화
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = threshold
        self.n_steps = n_steps
        
        # syn_a 초기화
        self.register_buffer('syn_a', self.init_syn_a(n_steps, tau_s))
        self.register_buffer('membrain_input', None)
        
    def init_syn_a(self, n_steps, tau_s):
        syn_a = torch.zeros((1, 1, 1, 1, n_steps), dtype=torch.float32)
        syn_a[..., 0] = 1
        for t in range(n_steps - 1):
            syn_a[..., t + 1] = (
                syn_a[..., t] - syn_a[..., t] / tau_s
            )
        syn_a /= tau_s
        return syn_a
    
    def extra_repr(self):
        org_repr = super(SNNLinear, self).extra_repr()
        
        return f"{org_repr}, use_tsslbp={self.use_tsslbp}, tau_m={self.tau_m}, tau_s={self.tau_s}, threshold={self.threshold}, n_steps={self.n_steps}"
    
    def forward(self, x):
        # 입력 차원 조정
        x = x.view(x.shape[0], -1, x.shape[-1])  # (batch_size, in_features, n_steps)
        x = x.transpose(1, 2)
        y = F.linear(x, self.weight, None)  # bias는 None
        
        # x와 동일한 device와 dtype으로 syn_a 복사
        device_syn_a = self.syn_a.to(x.device).type(x.dtype)
        
        # 출력 차원 복원
        y = y.transpose(1, 2)
        y = y.view(y.shape[0], self.out_features, 1, 1, -1)
        
        self.membrain_input = y
        
        if self.use_tsslbp:
            y = TSSLBP.apply(y, self.tau_m, self.tau_s, self.threshold, device_syn_a)
            
        return y

    def weight_clipper(self, clip_value=4):
        with torch.no_grad():
            self.weight.clamp_(-clip_value, clip_value)

class SNNDropout(nn.Module):
    def __init__(self, p = 0.5, inplace=False):
        super(SNNDropout, self).__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x):
        return F.dropout3d(x, self.p, self.training, self.inplace)
    
class SNNSumPooling(nn.Conv3d):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel_size = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                "kernelSize can only be of 1 or 2 dimension. It was: {}".format(
                    kernel_size.shape
                )
            )

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                "stride can be either int or tuple of size 2. It was: {}".format(
                    stride.shape
                )
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                "padding can be either int or tuple of size 2. It was: {}".format(
                    padding.shape
                )
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                "dilation can be either int or tuple of size 2. It was: {}".format(
                    dilation.shape
                )
            )
        super(SNNSumPooling, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False  # Conv3d 생성자에 bias=False 명시
        )
        
        # weight initialization
        self.weight = torch.nn.Parameter(1 * torch.ones(self.weight.shape), requires_grad=False)

    def forward(self, x):
        out = F.conv3d(
            x.reshape(x.shape[0], 1, x.shape[1]*x.shape[2], x.shape[3], x.shape[4]),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )
        
        return out.reshape(out.shape[0], x.shape[1], -1, out.shape[3], out.shape[4])
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from eva.datasets import PropheseeGen4CLS

from eva.eva import EVA

from eva import transforms as eva_transforms
from eva import utils, model, loss
import torch

    
if __name__ == "__main__":
    utils.set_seed(7)
    
    # # load dataset
    transform = transforms.Compose([
        eva_transforms.RandomTemporalCrop(time_window = 66_000, padding_enable=True),
        # transforms.Denoise(filter_time=11000),
        eva_transforms.ToFrameAuto(n_time_bins=5, aspect_ratio=False),
        eva_transforms.MergeFramePolarity(),
        # eva_transforms.EventFrameResize((64,64)),
        # eva_transforms.EventFrameRandomResizedCrop((64,64)),
        eva_transforms.EventFrameRandomResizedCrop((80,80)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    train_ds = PropheseeGen4CLS(root = '/data/event-data/Prophesee_Gen4CLS_99_mini', train=True, transform=transform)
    
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 66_000, padding_enable=True),
        # transforms.Denoise(filter_time=11000),
        eva_transforms.ToFrameAuto(n_time_bins=5, aspect_ratio=False),
        eva_transforms.MergeFramePolarity(),
        # eva_transforms.EventFrameResize((64,64)),
        eva_transforms.EventFrameResize((80,80)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    val_ds = PropheseeGen4CLS(root = '/data/event-data/Prophesee_Gen4CLS_99_mini', train=False, transform=transform)
    
    batch_size = 16
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    # snn_model = model.PPGen4NetMini(16, 1, n_steps = 5)
    snn_model = model.PPGen4NetMini128(16, 1, n_steps = 5)
    print(snn_model.summary(input_shape = (5,1,80,80)))

    optimizer = torch.optim.AdamW(snn_model.parameters(), lr=0.0005)
    spikeloss = loss.SpikeCountLoss(5, 0, focal=True)
    
    num_epochs = 100  # 원하는 에포크 수 설정
    
    snn_eva = EVA(snn_model, optimizer, spikeloss, (train_loader, val_loader), num_epochs, "cuda:0")
    snn_eva.trainer()
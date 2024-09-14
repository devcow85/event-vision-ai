import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from eva.eva import EVA

from eva import transforms as eva_transforms
from eva import utils, model, loss
import torch

if __name__ == "__main__":
    utils.set_seed(7)
    
    # load dataset
    transform = transforms.Compose([
        eva_transforms.RandomTemporalCrop(time_window = 99000),
        transforms.Denoise(filter_time=11000),
        transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,
                           n_time_bins=5)
    ])
    
    train_ds = tonic.datasets.DVSGesture(save_to = '/data', 
                                         train=True, 
                                         transform = transform)
    
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 99000),
        transforms.Denoise(filter_time=11000),
        transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,
                           n_time_bins=5)
    ])
    
    val_ds = tonic.datasets.DVSGesture(save_to = '/data', 
                                         train=False, 
                                         transform = transform)
    
    batch_size = 64
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    snn_model = model.DVSGestureNet(5, 1, n_steps = 5)
    
    optimizer = torch.optim.AdamW(snn_model.parameters(), lr=0.0005)
    spikeloss = loss.SpikeCountLoss(4, 1)
    
    num_epochs = 100  # 원하는 에포크 수 설정
    
    snn_eva = EVA(snn_model, optimizer, spikeloss, (train_loader, val_loader), num_epochs, "cuda:0")
    snn_eva.trainer()
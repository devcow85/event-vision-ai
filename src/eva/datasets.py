from torch.utils.data import DataLoader

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset


def get_gesture_dataset(data_path, batch_size, n_steps, dt):
    
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    
    transform = transforms.Compose([
        transforms.Denoise(filter_time=dt),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps)
    ])
    
    train_ds = tonic.datasets.DVSGesture(save_to = data_path, train=True, transform=transform)
    val_ds = tonic.datasets.DVSGesture(save_to = data_path, train=False, transform=transform)
    
    train_cache_ds = DiskCachedDataset(train_ds, cache_path="./cache/fast_train_dataloading")
    val_cache_ds = DiskCachedDataset(val_ds, cache_path="./cache/fast_val_dataloading")

    train_loader = DataLoader(train_cache_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_cache_ds, shuffle=False, batch_size=batch_size)
    
    return train_loader, val_loader

def get_nmnist_dataset(data_path, batch_size, n_steps, dt):
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=dt),
            transforms.ToFrame(
                sensor_size=sensor_size, n_time_bins=n_steps
            ),
        ]
    )
    
    train_ds = tonic.datasets.NMNIST(save_to = data_path, train=True, transform=transform)
    val_ds = tonic.datasets.NMNIST(save_to = data_path, train=False, transform=transform)
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader
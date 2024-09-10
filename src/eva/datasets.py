from torch.utils.data import DataLoader

import tonic
import tonic.transforms as transforms

def get_gesture_dataset(data_path, batch_size, n_steps, dt):
    
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    
    transform = transforms.Compose([
        transforms.Denoise(filter_time=dt),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_steps)
    ])
    
    train_ds = tonic.datasets.DVSGesture(save_to = data_path, train=True, transform=transform)
    val_ds = tonic.datasets.DVSGesture(save_to = data_path, train=False, transform=transform)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
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
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    return train_loader, val_loader
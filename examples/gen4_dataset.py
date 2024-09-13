from eva import datasets
import tonic.transforms as transforms
from eva import transforms as eva_transforms
from eva.datasets import PropheseeGen4mini
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np

import time

def normalize_and_convert_to_uint8(data):
    """데이터를 0-255 사이로 정규화하고, uint8로 변환"""
    data_min = np.min(data)
    data_max = np.max(data)
    # 데이터가 이미 동일한 값일 경우 0으로 나눌 수 없으니, 이 경우 대비
    if data_max != data_min:
        normalized_data = (data - data_min) / (data_max - data_min) * 255.0
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data.astype(np.uint8)

def plot_images(event_data):
    data = event_data[0]
    target = event_data[1]
    
    print(target)
    for idx, evt_frame in enumerate(data):
        plt.imshow(normalize_and_convert_to_uint8(evt_frame[0]))
        plt.savefig(f"test_{idx}.png")
    # print("img_shape", img.shape)
    # i8 = normalize_and_convert_to_uint8(img)
    # plt.imsave('examples/onefile.png', i8)
    
if __name__ == "__main__":
    print('load dataset')
    # prophesee_gen4_mini
    # Prophesee_Dataset_n_cars
    
    # # load dataset
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 99_000),
        # transforms.Denoise(filter_time=11000),
        transforms.ToFrame(sensor_size=PropheseeGen4mini.sensor_size, n_time_bins=5),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventFrameResize((256,128)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    train_ds = PropheseeGen4mini(root = '/data/prophesee_gen4_mini', train=True, transform=transform)
    
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 99_000),
        # transforms.Denoise(filter_time=11000),
        transforms.ToFrame(sensor_size=PropheseeGen4mini.sensor_size, n_time_bins=5),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventFrameResize((256,128)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    val_ds = PropheseeGen4mini(root = '/data/prophesee_gen4_mini', train=False, transform=transform)
    
    batch_size = 16
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    # for idx, (data, targets) in enumerate(train_loader):
    #     print(data.shape, targets)
    # import os
    for idx, (data, target) in enumerate(train_ds):
        print(data.shape, target, train_ds.file_path[idx])
        
        if data.size == 0:
            print(data.shape, target, train_ds.file_path[idx])
            # os.remove(train_ds.file_path[idx])
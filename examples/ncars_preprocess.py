import tonic.transforms as transforms
from torch.utils.data import DataLoader
from eva import transforms as eva_transforms
from eva import datasets

import numpy as np
from PIL import Image

import os


if __name__ == "__main__":
    # load dataset
    transform = transforms.Compose([
        eva_transforms.RandomTemporalCrop(time_window = 66000),
        eva_transforms.ToFrameAuto(n_time_bins=5),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.MinMaxScaler(min_val=100, max_val=260),
        eva_transforms.EventFrameResize((64,64))
    ])
    
    train_ds = datasets.NCARS(root = '/data/Prophesee_Dataset_n_cars', train=True, transform=transform)
    
    transform = transforms.Compose([
        eva_transforms.TemporalCrop(time_window = 99000),
        transforms.Denoise(filter_time=11000),
        eva_transforms.ToFrameAuto(n_time_bins=5),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventFrameResize((64,64)),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
    ])
    
    val_ds = datasets.NCARS(root = '/data/Prophesee_Dataset_n_cars', train=False, transform=transform)
    
    # train_loader = DataLoader(train_ds, shuffle=False, batch_size=128, num_workers=8)
    
    # for idx, (data, label) in enumerate(train_loader):
    #     print(data.shape)

    # glv_min, glv_max = 3000, -3000
    for idx, (data, label) in enumerate(train_ds):
        # if glv_min > data.min():
        #     glv_min = data.min()
        
        # if glv_max < data.max():
        #     glv_max = data.max()
        # print(glv_min, glv_max)
        
        concatenated_image = np.concatenate(data, axis=1)
        final_image = Image.fromarray(concatenated_image[0].astype(np.uint8))
        
        filename = list(train_ds.file_paths.items())[idx]
        
        os.makedirs('data/ncars/train',exist_ok=True)
        final_image.save(f'data/ncars/train/{os.path.basename(filename[0]).split(".")[0]}.png')

        # print(concatenated_image.shape, filename)
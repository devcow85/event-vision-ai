import tonic
from eva import transforms as eva_transforms
import tonic.transforms as transforms

import time
import matplotlib.pyplot as plt

import numpy as np

def plot_images(event_data):
    data = event_data[0]
    target = event_data[1]
    
    img = np.concatenate([np.stack([evd[0], evd[1], np.zeros(evd[0].shape)], -1) for evd in data], 1)
    i8 = normalize_and_convert_to_uint8(img)
    plt.imsave('examples/onefile.png', i8)

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

if __name__ == "__main__":
    
    transform = transforms.Compose([
        eva_transforms.RandomTemporalCrop(time_window = 132000),
        eva_transforms.ToFrameAuto(n_time_bins=5)
    ])
    
    train_ds = tonic.datasets.DVSGesture(save_to = '/data', 
                                         train=True, 
                                         transform = transform)
    
    start_time = time.time()
    print(train_ds[0][0].shape)
    print('elapsed', time.time() - start_time)
    
    plot_images(train_ds[0])
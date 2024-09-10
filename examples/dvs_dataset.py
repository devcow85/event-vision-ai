from tsslbp import datasets
import matplotlib.pyplot as plt

import numpy as np

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

def plot_images(loader, n):
    images_concat = []  # 각 이미지별로 수평으로 concat된 이미지를 저장할 리스트

    for i in range(min(n, len(loader.dataset))):
        data, label = loader.dataset[i]  # 데이터와 레이블을 가져옴
        frames, polarity, width, height = data.shape  # 데이터 모양 확인
        print(f"Image {i} - Shape: {data.shape}, Label: {label}")

        # 각 프레임을 수평으로 이어 붙이기 (polarity 0만 사용)
        frame_concat = np.concatenate([
            np.stack([data[j, 0], data[j, 1], np.zeros((width, height))], axis=-1)  # R=polarity 0, G=polarity 1, B=0
            for j in range(frames)
        ], axis=1)
        
        # 이어 붙인 이미지를 리스트에 저장
        images_concat.append(frame_concat)

    # 수평으로 이어 붙인 이미지들을 다시 수직으로 이어 붙임
    final_image = np.concatenate(images_concat, axis=0)
    i8_data = normalize_and_convert_to_uint8(final_image)
    
    plt.imsave('train_concat_image.png', i8_data)
       
if __name__ == "__main__":
    print('load dataset')
    
    train_loader, val_loader = datasets.get_nmnist_dataset('data/', 64, 10, 1000)
    
    plot_images(train_loader, 10)
    
    

from eva import datasets
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
    
    plt.imsave('examples/train_concat_image.png', i8_data)

def load_dataset(loader, n):
    loading_time = []
    
    loop_start = time.time()
    for i in range(n):  # Use n as the loop count
        start_time = time.time()
        events, target = loader.dataset[i]
        loading_time.append(time.time() - start_time)
    loop_end = time.time()
    
    print('total loop time', loop_end - loop_start)
    
    
    # Calculate the mean and standard deviation
    mean_time = np.mean(loading_time)
    std_dev = np.std(loading_time)
    
    # Convert times to seconds and milliseconds for formatting
    mean_time_s = mean_time * 1000
    std_dev_ms = std_dev * 1000  # Convert to milliseconds
    
    # Print in the desired format
    print(f"{mean_time_s:.2f} ms ± {std_dev_ms:.0f} ms per image (mean ± std. dev. of {n} runs, 1 loop each)")

def load_dataloader(loader, n):
    loading_time = []
    
    dlen = 0
    loop_start = time.time()
    # Loop through the loader with n batches (or iterations)
    for idx, (events, target) in enumerate(loader):
        dlen+=events.shape[0]
        if idx >= n:  # Stop after n iterations
            break
        start_time = time.time()
        
        # Access the batch data (events and target are the loaded data)
        _ = events, target  # Simulating data usage without actually processing
        
        loading_time.append((time.time() - start_time)/dlen)
        
    loop_end = time.time()
    print('total loop time', loop_end - loop_start)
    
    # Calculate the mean and standard deviation
    mean_time = np.mean(loading_time)
    std_dev = np.std(loading_time)
    
    # Convert times to seconds and milliseconds for formatting
    mean_time_s = mean_time * 1000 
    std_dev_ms = std_dev * 1000  # Convert to milliseconds
    
    # Print in the desired format
    print(f"{mean_time_s:.2f} ms ± {std_dev_ms:.0f} ms per image (mean ± std. dev. of {n} runs, 1 loop each)")


if __name__ == "__main__":
    print('load dataset')
    
    train_loader, val_loader = datasets.get_gesture_dataset('/data', 64, 10, 1000)
    
    plot_images(train_loader, 10)
    
    load_dataset(train_loader, 640)
    load_dataloader(train_loader, 10)
    

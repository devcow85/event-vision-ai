from eva import transforms as eva_transforms
from eva import datasets
import tonic.transforms as transforms
import numpy as np

import numpy as np
import cv2

ds = datasets.PropheseeGen4CLS(root = '/data/event-data/Prophesee_Gen4CLS_99_mini', train=False)


event_data = ds[80][0]

import matplotlib.pyplot as plt

preprocessing = transforms.Compose([
        eva_transforms.ToFrameAuto(n_time_bins=1),
        eva_transforms.MergeFramePolarity(),
        eva_transforms.EventNormalize(mean=(128,), std=(1,))
])

img = preprocessing(event_data)[0][0]

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # X축 Sobel 필터
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Y축 Sobel 필터
sobel_edge = np.sqrt(sobel_x**2 + sobel_y**2)  # 엣지 강도 계산

# 플롯 생성: 원본 이미지, 강화된 이미지, 각 이벤트의 X 히스토그램
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# 원본 이미지
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title("Original Event Frame")

# 원본 이벤트 데이터의 X 좌표 히스토그램
ax[1, 0].hist(img, bins=50, alpha=0.7)
ax[1, 0].set_title('Original X Histogram')

# 강화된 이미지
ax[0, 1].imshow(sobel_edge, cmap='gray')
ax[0, 1].set_title("Spatiotemporal Event Enhancement Frame")

# 강화된 이벤트 데이터의 X 좌표 히스토그램
ax[1, 1].hist(sobel_edge, bins=50, alpha=0.7)
ax[1, 1].set_title('Enhanced X Histogram')

plt.tight_layout()
plt.show()
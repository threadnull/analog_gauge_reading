import torch
import numpy as np
import matplotlib.pyplot as plt

# 가우시안 히트맵
def gaussian_heatmap(keypoints, input_size=640, output_size=160, sigma=2):
    num_kp = len(keypoints)
    # [키포인트 개수, 160, 160]
    heatmaps = np.zeros((num_kp, output_size, output_size), dtype=np.float32)
    # 입출력 비율(4배)
    stride = input_size / output_size 

    for i in range(num_kp):
        # 좌표 스케일링 (640 -> 160)
        mu_x = keypoints[i][0] / stride
        mu_y = keypoints[i][1] / stride
        
        # 고속 배열 연산
        y, x = np.ogrid[0:output_size, 0:output_size]
        dist_sq = (x - mu_x)**2 + (y - mu_y)**2
        heatmaps[i] = np.exp(-dist_sq / (2 * sigma**2))

    return torch.from_numpy(heatmaps)

def visualize_dataset_sample(dataset, index):
    # 1. 데이터셋에서 샘플 가져오기
    image, heatmaps = dataset[index]
    
    # 2. PyTorch 텐서를 NumPy 배열로 변환 (시각화를 위해)
    if isinstance(image, torch.Tensor):
        # (C, H, W) -> (H, W, C)
        img_to_show = image.permute(1, 2, 0).numpy()
    else:
        img_to_show = image
    
    # 히트맵은 모든 채널(키포인트)을 하나로 합쳐서 시각화하거나 개별로 볼 수 있습니다.
    # 여기서는 모든 히트맵을 최대값 기준으로 합칩니다.
    combined_heatmap = np.max(heatmaps.numpy(), axis=0)

    # 3. 시각화 설정
    plt.figure(figsize=(12, 5))

    # --- 변화된 부분: 왼쪽에는 원본 이미지, 오른쪽에는 히트맵 오버레이 표시 ---
    
    # # 첫 번째: 원본 이미지
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(img_to_show)
    # plt.axis('off')

    # 두 번째: 원본 이미지 위에 히트맵 겹치기
    plt.subplot(1, 2, 2)
    plt.title(f"{index} Heatmap")
    plt.imshow(img_to_show) # 배경에 원본 이미지
    # 히트맵의 크기를 원본 이미지 크기(640x640)에 맞게 확대하여 겹침
    plt.imshow(combined_heatmap, cmap='jet', alpha=0.5, extent=(0, 640, 640, 0)) 
    plt.axis('off')

    plt.tight_layout()
    plt.show()
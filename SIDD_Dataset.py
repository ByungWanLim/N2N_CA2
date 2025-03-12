import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class SIDD_Dataset(Dataset):
    def __init__(self, noisy, gt, img_size=512, cache_dir='./dataset_cache'):
        """
        SIDD 데이터셋 클래스 (완전 캐싱 기능 포함)
        """
        print(f"=== SIDD_Dataset 초기화 (img_size={img_size}) ===")
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        # 빠른 데이터 해시값 계산 (캐시 키로 사용)
        data_id = f"sidd_processed_{img_size}"
        cache_file = os.path.join(cache_dir, f'{data_id}.pt')
        
        # 캐시 파일이 존재하는지 확인
        if os.path.exists(cache_file):
            print(f"캐시 파일을 로드합니다: {cache_file}")
            start_time = time.time()
            cache_data = torch.load(cache_file)
            self.img1_all = cache_data['img1']
            self.img2_all = cache_data['img2']
            self.img3_all = cache_data['img3']
            self.downsampled_gt = cache_data['downsampled_gt']
            print(f"캐시 로드 완료. 소요 시간: {time.time() - start_time:.2f}초")
            print(f"데이터 크기: {len(self.img1_all)} 샘플, 이미지 크기: {self.img1_all.shape[2]}x{self.img1_all.shape[3]}")
        else:
            print("데이터셋을 처리하고 캐싱합니다...")
            start_time = time.time()
            
            # 데이터 전처리 및 리사이징
            if np.max(noisy) > 1.0:
                noisy = noisy.astype(np.float32) / 255.0
                gt = gt.astype(np.float32) / 255.0
                
            # 텐서 변환 및 채널 순서 변경
            noisy_tensor = torch.tensor(noisy, dtype=torch.float32).permute(0, 3, 1, 2)
            gt_tensor = torch.tensor(gt, dtype=torch.float32).permute(0, 3, 1, 2)
            
            # Resize
            transform = transforms.Resize((img_size, img_size), 
                                        interpolation=transforms.InterpolationMode.BILINEAR,
                                        antialias=True)
            noisy_resized = transform(noisy_tensor)
            gt_resized = transform(gt_tensor)
            
            # 짝수 크기로 조정
            c, h, w = noisy_resized.shape[1], noisy_resized.shape[2], noisy_resized.shape[3]
            if h % 2 != 0:
                noisy_resized = noisy_resized[:, :, :h-1, :]
                gt_resized = gt_resized[:, :, :h-1, :]
            if w % 2 != 0:
                noisy_resized = noisy_resized[:, :, :, :w-1]
                gt_resized = gt_resized[:, :, :, :w-1]
            
            # 다운샘플링된 GT
            self.downsampled_gt = F.avg_pool2d(gt_resized, kernel_size=2, stride=2)
            
            # 출력 크기 계산
            n_samples = len(noisy_resized)
            c = noisy_resized.shape[1]
            out_h, out_w = h // 2, w // 2
            
            # 모든 샘플을 미리 처리 (실제 최적화 부분!)
            print(f"모든 샘플에 대해 2x2 샘플링을 미리 처리합니다...")
            self.img1_all = torch.zeros((n_samples, c, out_h, out_w), dtype=torch.float32)
            self.img2_all = torch.zeros((n_samples, c, out_h, out_w), dtype=torch.float32)
            self.img3_all = torch.zeros((n_samples, c, out_h, out_w), dtype=torch.float32)
            
            # 모든 샘플에 대해 픽셀 샘플링 미리 계산
            for idx in tqdm(range(n_samples)):
                noisy_img = noisy_resized[idx]
                
                # 각 2x2 커널을 순회
                for y in range(0, h, 2):
                    for x in range(0, w, 2):
                        # 2x2 블록이 이미지 경계를 넘어가는지 확인
                        if y + 1 >= h or x + 1 >= w:
                            continue
                        
                        # 현재 2x2 블록의 픽셀
                        positions = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
                        
                        # 무작위로 2개 위치 선택
                        selected_positions = random.sample(positions, 2)
                        remaining_positions = [p for p in positions if p not in selected_positions]
                        
                        out_y, out_x = y // 2, x // 2
                        
                        # 선택된 픽셀 저장
                        self.img1_all[idx, :, out_y, out_x] = noisy_img[:, selected_positions[0][0], selected_positions[0][1]]
                        self.img2_all[idx, :, out_y, out_x] = noisy_img[:, selected_positions[1][0], selected_positions[1][1]]
                        
                        # 나머지 픽셀의 평균 저장
                        avg_pixel = torch.zeros(c, dtype=torch.float32)
                        for y_pos, x_pos in remaining_positions:
                            avg_pixel += noisy_img[:, y_pos, x_pos]
                        avg_pixel /= 2
                        self.img3_all[idx, :, out_y, out_x] = avg_pixel
            
            # 값 범위 클리핑
            self.img1_all = torch.clamp(self.img1_all, 0.0, 1.0)
            self.img2_all = torch.clamp(self.img2_all, 0.0, 1.0)
            self.img3_all = torch.clamp(self.img3_all, 0.0, 1.0)
            
            # 처리된 모든 데이터 캐싱
            print(f"처리된 데이터를 캐시에 저장합니다: {cache_file}")
            torch.save({
                'img1': self.img1_all,
                'img2': self.img2_all,
                'img3': self.img3_all,
                'downsampled_gt': self.downsampled_gt
            }, cache_file)
            
            print(f"모든 처리 완료. 소요 시간: {time.time() - start_time:.2f}초")
    
    def __len__(self):
        return len(self.img1_all)
    
    def __getitem__(self, idx):
        """
        매우 빠른 데이터셋 조회 - 미리 계산된 결과만 반환
        """
        return self.img1_all[idx], self.img2_all[idx], self.img3_all[idx], self.downsampled_gt[idx]
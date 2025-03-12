# 필요한 라이브러리 import
from thop import profile
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# PSNR calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 시각화 함수
def visualize_input_target_output(model, dataloader, device, num_samples=2):
    """모델의 입력, 타겟, 출력을 시각화합니다"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(16, 4*num_samples))
    
    with torch.no_grad():
        for i, (input1, input2, targets, cleans) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            input1 = input1.to(device)
            input2 = input2.to(device)
            targets = targets.to(device)
            cleans = cleans.to(device)
            
            # 모델 예측
            inputs = torch.cat((input1, input2), dim=1)
            outputs = model(inputs)
            
            # 시각화
            # 입력 이미지 1 (채널 0-2)
            axes[i, 0].imshow(input1[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, 0].set_title('Input Image 1')
            axes[i, 0].axis('off')
            
            # 입력 이미지 2 (채널 3-5)
            axes[i, 1].imshow(input2[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, 1].set_title('Input Image 2')
            axes[i, 1].axis('off')
            
            # 타겟 이미지
            axes[i, 2].imshow(targets[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, 2].set_title('Target Image')
            axes[i, 2].axis('off')
            
            # 깨끗한 원본 이미지
            axes[i, 3].imshow(cleans[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, 3].set_title('Clean Original Image')
            axes[i, 3].axis('off')
            
            # 출력 이미지
            axes[i, 4].imshow(outputs[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            
            # PSNR 계산
            mse = F.mse_loss(outputs[0], targets[0]).item()
            psnr = -10 * np.log10(mse) if mse > 0 else 100
            axes[i, 4].set_title(f'Output Image (PSNR: {psnr:.2f}dB)')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.show()

# 모델 프로파일링
def measure_model_complexity(model, img_size=256, device='cuda'):
    """
    모델의 파라미터 수와 GFLOPs 계산
    
    Args:
        model: 파라미터 수와 GFLOPs를 계산할 모델
        
    Returns:
        tuple: (파라미터 수, GFLOPs)
    """
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    input_sample = torch.randn(1, 6, img_size, img_size).to(device)  # CIFAR-10 이미지 크기 (32x32)
    flops, _ = profile(model, inputs=(input_sample,))
    flops = flops / 1e9  # GFLOPs로 변환
    return params, flops
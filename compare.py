# test.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.DnCNN import DnCNN
from models.DnCNN_CA import DnCNN_CA
from utils import calculate_psnr
from SIDD_Dataset import SIDD_Dataset

def load_model(model_type='standard', device='cuda'):
    """
    체크포인트에서 모델 불러오기
    
    Args:
        model_path: 모델 체크포인트 경로
        model_type: 'standard' 또는 'ca'
        device: 'cuda' 또는 'cpu'
    
    Returns:
        학습된 모델
    """
    if model_type == 'ca':
        model = DnCNN_CA(in_channels=6, out_channels=3)
    else:
        model = DnCNN(in_channels=6, out_channels=3)
    
    model_path = f'./checkpoints/final_dncnn_model{"_ca" if model_type == "ca" else ""}.pth'

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print(f"{model_path} 모델 불러오기 완료")
    
    return model

def evaluate_model(model, val_loader, device='cuda'):
    """
    검증 데이터에서 모델 성능 평가
    
    Returns:
        평균 PSNR
    """
    model.eval()
    total_psnr = 0
    count = 0
    
    with torch.no_grad():
        for img1, img2, img3, targets in tqdm(val_loader, desc="모델 평가 중"):
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            
            # 입력 이미지 결합
            inputs = torch.cat((img1, img2), dim=1)
            
            # 모델 예측
            outputs = model(inputs)
            
            # PSNR 계산
            for i in range(outputs.size(0)):
                psnr = calculate_psnr(outputs[i], targets[i])
                total_psnr += psnr
                count += 1
    
    avg_psnr = total_psnr / count
    return avg_psnr

def visualize_results(models, model_names, val_loader, device='cuda', num_samples=2):
    """
    여러 모델의 결과 시각화 및 비교
    
    Args:
        models: 모델 리스트
        model_names: 모델 이름 리스트
        val_loader: 검증 데이터 로더
        device: 'cuda' 또는 'cpu'
        num_samples: 시각화할 샘플 수
    """
    for model in models:
        model.eval()
    
    # 데이터셋에서 샘플 선택
    data_iter = iter(val_loader)
    img1, img2, img3, targets = next(data_iter)
    
    # 결과 플롯 설정
    fig, axes = plt.subplots(num_samples, len(models) + 2, figsize=(4 * (len(models) + 2), 4 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            # 원본 이미지 표시
            axes[i, 0].imshow(img1[i].permute(1, 2, 0).numpy())
            axes[i, 0].set_title('Input 1')
            axes[i, 0].axis('off')
            
            # 타겟 이미지 표시
            axes[i, -1].imshow(targets[i].permute(1, 2, 0).numpy())
            axes[i, -1].set_title('Ground Truth')
            axes[i, -1].axis('off')
            
            # 각 모델의 예측 표시
            for j, (model, name) in enumerate(zip(models, model_names)):
                # 입력 이미지 결합
                input_i = torch.cat((img1[i:i+1].to(device), img2[i:i+1].to(device)), dim=1)
                
                # 예측
                output = model(input_i)
                output_np = output[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                
                # PSNR 계산
                psnr = calculate_psnr(output[0].cpu(), targets[i])
                
                # 결과 표시
                axes[i, j + 1].imshow(output_np)
                axes[i, j + 1].set_title(f'{name} (PSNR: {psnr:.2f}dB)')
                axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./checkpoints/model_comparison.png', dpi=300)
    plt.show()

def compare_models(model_types, model_names, val_loader, device='cuda'):
    """
    여러 모델의 성능 비교
    
    Args:
        model_paths: 모델 체크포인트 경로 리스트
        model_types: 모델 유형 리스트 ('standard' 또는 'ca')
        model_names: 모델 이름 리스트
        val_loader: 검증 데이터 로더
        device: 'cuda' 또는 'cpu'
    
    Returns:
        모델 리스트, PSNR 리스트
    """
    models = []
    psnrs = []
    
    # 모델 로드 및 평가
    for type_name in model_types:
        print(f"DnCNN{"_ca" if type_name == "ca" else ""} 모델 평가 중...")
        model = load_model(model_type=type_name, device=device)
        models.append(model)
        
        psnr = evaluate_model(model, val_loader, device)
        psnrs.append(psnr)
        print(f"DnCNN{"_ca" if type_name == "ca" else ""} 평균 PSNR: {psnr:.2f}dB")
    
    # PSNR 비교 그래프
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, psnrs, color=['blue', 'orange', 'green', 'red'][:len(models)])
    
    plt.ylabel('PSNR (dB)')
    plt.title('모델별 PSNR 비교')
    for bar, psnr in zip(bars, psnrs):
        plt.text(bar.get_x() + bar.get_width()/2, psnr + 0.1, f"{psnr:.2f}dB", 
                 ha='center', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('./checkpoints/psnr_comparison.png', dpi=300)
    plt.show()
    
    return models, psnrs

def run_evaluation(img_size=128, batch_size=4):
    """
    모델 평가 실행 함수
    
    Args:
        dataset: SIDD 데이터셋 객체
        img_size: 이미지 크기
        batch_size: 배치 크기
    """
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 디바이스: {device}")
    
    # 학습/검증 분할
    data_path = 'data'
    noisy_data = np.load(f'{data_path}/noisy_img_array.npy', mmap_mode='r')
    gt_data = np.load(f'{data_path}/gt_img_array.npy', mmap_mode='r')
    noisy_data = noisy_data.astype(np.float32) / 255.0
    gt_data = gt_data.astype(np.float32) / 255.0

    dataset = SIDD_Dataset(noisy_data, gt_data, img_size=img_size)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 검증 데이터로더 생성
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    # 모델 경로 및 정보 설정
    model_types = ['standard', 'ca']
    model_names = ['DnCNN', 'DnCNN-CA']
    
    # 모델 평가 및 비교
    models, psnrs = compare_models(model_types, model_names, val_loader, device)
    
    # 결과 시각화
    visualize_results(models, model_names, val_loader, device, num_samples=2)
    
    # return psnrs
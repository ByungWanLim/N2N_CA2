import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import visualize_input_target_output, calculate_psnr


# 학습 함수
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stop, ca=None):
    """
    DnCNN 모델 학습 함수
    
    Args:
        model: DnCNN 모델
        train_loader: 학습 데이터로더
        val_loader: 검증 데이터로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        num_epochs: 에폭 수
        device: 학습 디바이스 (cuda 또는 cpu)
    """
    model.to(device)
    best_val_loss = float('inf')
    
    # 학습 기록
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        # 학습
        for input1, input2, targets, cleans in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input1, input2, targets, cleans = input1.to(device), input2.to(device), targets.to(device), cleans.to(device)
            inputs = torch.cat((input1, input2), dim=1)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            
            # 가중치 업데이트
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item() * inputs.size(0)
        
        # 에폭 평균 손실
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for input1, input2, targets, cleans in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input1, input2, targets, cleans = input1.to(device), input2.to(device), targets.to(device), cleans.to(device)
                inputs = torch.cat((input1, input2), dim=1)
                
                # 순전파
                outputs = model(inputs)
                
                # 손실 계산
                loss = criterion(outputs, targets)
                
                # 손실 누적
                val_loss += loss.item() * inputs.size(0)
                
                for i in range(outputs.size(0)):
                    # outputs와 cleans 비교 (targets 대신)
                    psnr = calculate_psnr(outputs[i], cleans[i])
                    val_psnr += psnr.item()  # 텐서를 스칼라로 변환
        
        # 에폭 평균 손실 및 PSNR
        val_loss = val_loss / len(val_loader.dataset)
        val_psnr = val_psnr / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val PSNR: {val_psnr:.2f}dB")
        
        # 모델 저장 (가장 좋은 검증 손실)
        if ca is not None:
            model_name = f'best_unet_model_{ca}.pth'
        else:
            model_name = f'best_unet_model.pth'
        model_path = os.path.join('./checkpoints', model_name)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"모델 저장됨 (Val Loss: {val_loss:.4f})")
        else:
            patient += 1
        
        # 조기 종료
        if early_stop is not None:
            if patient >= early_stop:
                print(f"조기 종료 ({early_stop} 에폭 동안 검증 손실이 감소하지 않음)")
                break
        
        # 중간 결과 시각화 (5에폭마다)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            visualize_input_target_output(model, val_loader, device, num_samples=2)
    
    # 학습 곡선 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return train_losses, val_losses
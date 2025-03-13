import torch
import torch.nn as nn
import torch.nn.functional as F


# DnCNN 모델 정의
class DnCNN(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_layers=17, features=64):
        """
        DnCNN 모델 (2개의 입력 이미지를 concat하여 1개의 타겟 이미지 예측)
        
        Args:
            in_channels (int): 입력 채널 수 (2개의 RGB 이미지 concat -> 6)
            out_channels (int): 출력 채널 수 (RGB 이미지 -> 3)
            num_layers (int): 레이어 수
            features (int): 특징 맵 수
        """
        super(DnCNN, self).__init__()
        
        layers = []
        
        # 첫 번째 레이어: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # 중간 레이어: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # 마지막 레이어: Conv
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (B, 6, H, W) - 2개의 RGB 이미지 concat
            
        Returns:
            출력 이미지 텐서 (B, 3, H, W)
        """
        return self.dncnn(x)
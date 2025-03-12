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
    
# DnCNN 모델 정의
class DnCNN_CA(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_layers=15, features=64):
        """
        DnCNN 모델 (2개의 입력 이미지를 concat하여 1개의 타겟 이미지 예측)
        
        Args:
            in_channels (int): 입력 채널 수 (2개의 RGB 이미지 concat -> 6)
            out_channels (int): 출력 채널 수 (RGB 이미지 -> 3)
            num_layers (int): 레이어 수
            features (int): 특징 맵 수
        """
        super(DnCNN_CA, self).__init__()
        
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
        # layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))
        
        self.dncnn = nn.Sequential(*layers)
        
        # Cross Attention을 위한 변환 레이어
        self.q_proj = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.k_proj = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.v_proj = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        
        # 출력 조정을 위한 레이어
        self.out_proj = nn.Conv2d(features // 2 , features, kernel_size=1)
        
        
        
        self.last_layer = nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (B, 6, H, W) - 2개의 RGB 이미지 concat
            
        Returns:
            출력 이미지 텐서 (B, 3, H, W)
        """
        x = self.dncnn(x)
        
         # 채널 축으로 반으로 나눔
        features_half = x.shape[1] // 2
        q = x[:, :features_half, ...]  # (B, features/2, H, W)
        k = x[:, features_half:, ...]  # (B, features/2, H, W)
        
        # Q, K, V 변환
        q = self.q_proj(q)  # (B, features/2, H, W)
        k = self.k_proj(k)  # (B, features/2, H, W)
        v = self.v_proj(k)  # (B, features/2, H, W)
        
        # 어텐션 계산을 위한 형태 변환
        batch_size, C, height, width = q.shape
        q = q.view(batch_size, C, height * width).permute(0, 2, 1)  # (B, H*W, C)
        k = k.view(batch_size, C, height * width)  # (B, C, H*W)
        v = v.view(batch_size, C, height * width).permute(0, 2, 1)  # (B, H*W, C)
        
        # 어텐션 가중치 계산
        x = torch.matmul(q, k) / (C ** 0.5)  # (B, H*W, H*W)
        x = F.softmax(x, dim=-1)
        
        # 어텐션 적용
        x = torch.matmul(x, v)  # (B, H*W, C)
        x = x.permute(0, 2, 1).view(batch_size, C, height, width)  # (B, C, H, W)
        
        # 출력 조정
        x = self.out_proj(x)  # (B, features, H, W)
        
        
        x = self.last_layer(x)
        
        return x
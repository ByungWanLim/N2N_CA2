import torch

import torch.nn as nn
import torch.nn.functional as F

class SpatialCrossAttention(nn.Module):
    def __init__(self, features):
        """
        이미지 데이터를 위한 Spatial Cross Attention 모듈
        
        Args:
            features (int): 특징 맵의 채널 수
        """
        super(SpatialCrossAttention, self).__init__()
        
        self.features = features
        feature_half = features // 2
        
        # Cross Attention을 위한 변환 레이어
        self.q_proj = nn.Conv2d(feature_half, feature_half, kernel_size=1)
        self.k_proj = nn.Conv2d(feature_half, feature_half, kernel_size=1)
        self.v_proj = nn.Conv2d(feature_half, feature_half, kernel_size=1)
        
        # 출력 조정을 위한 레이어
        self.input_proj = nn.Conv2d(features, feature_half, kernel_size=1)
        self.out_proj = nn.Conv2d(features, features, kernel_size=1)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 특징 맵 텐서 (B, features, H, W)
            
        Returns:
            출력 특징 맵 텐서 (B, features, H, W)
        """
        # 채널 축으로 반으로 나눔
        features_half = x.shape[1] // 2
        q = x[:, :features_half, ...]  # (B, features/2, H, W)
        k = x[:, features_half:, ...]  # (B, features/2, H, W)
        
        x = self.input_proj(x)  # (B, features, H, W)
        
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
        attn = torch.matmul(q, k) / (C ** 0.5)  # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        
        # 어텐션 적용
        out = torch.matmul(attn, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(batch_size, C, height, width)  # (B, C, H, W)
        
        out = torch.cat((x, out), dim=1)  # (B, features, H, W)
        
        # 출력 조정
        out = self.out_proj(out)  # (B, features, H, W)
        
        return out
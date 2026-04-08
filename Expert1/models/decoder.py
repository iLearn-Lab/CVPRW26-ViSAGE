

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.spatial_pool(x).squeeze(-1).squeeze(-1)
        attn = self.sigmoid(self.conv(attn))
        return attn.unsqueeze(-1).unsqueeze(-1)

class FeatureUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, target_size):
        x = self.proj(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x
class LearnableCenterBias(nn.Module):
    def __init__(self, h=224, w=224):
        super().__init__()
        self.cx = nn.Parameter(torch.tensor(0.0))
        self.cy = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.weight = nn.Parameter(torch.tensor(0.0)) 
        
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)

    def forward(self):
        gaussian = torch.exp(-((self.xx - self.cx)**2 + (self.yy - self.cy)**2) / (2 * self.sigma**2 + 1e-7))
        return self.weight * gaussian

class ResBlendBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)

class CVMMDecoder(nn.Module):
    def __init__(self, channel_list=[1024, 1024, 1024, 1024], embed_dim=256):
        super().__init__()
       
        c5, c11, c17, c23 = channel_list
        
       
        self.temp_attn = TemporalAttention(c23)
        self.proj23_attn = nn.Conv3d(c23, embed_dim, kernel_size=1) 
        
        self.upsample5 = FeatureUpsample(c5, embed_dim)
        self.upsample11 = FeatureUpsample(c11, embed_dim)
        self.proj17 = nn.Conv3d(c17, embed_dim, kernel_size=1)
        

        self.conv17 = ResBlendBlock3D(embed_dim, embed_dim)
        self.blend11 = ResBlendBlock3D(embed_dim * 2, embed_dim)
        self.blend5 = ResBlendBlock3D(embed_dim * 2, embed_dim)
        
        self.final_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=3, padding=1)
        )
        self.center_bias = LearnableCenterBias(h=224, w=224)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, gamma=None, beta=None):
        f5, f11, f17, f23 = features
        
        attn_weight = self.temp_attn(f23) 
        attn_map = self.proj23_attn(attn_weight) 
        
        feat17 = self.proj17(f17)
        feat17 = feat17 * attn_map 
        feat17 = self.conv17(feat17)
        target_size_11 = f11.shape[2:] 
        feat17_up = F.interpolate(feat17, size=target_size_11, mode='trilinear', align_corners=False)
        
        feat11 = self.upsample11(f11, target_size=target_size_11)
        feat11 = feat11 * attn_map 
        cat11 = torch.cat([feat11, feat17_up], dim=1) 
        blend11 = self.blend11(cat11)
        
        target_size_5 = f5.shape[2:]
        feat5 = self.upsample5(f5, target_size=target_size_5)
        feat5 = feat5 * attn_map 
        blend11_up = F.interpolate(blend11, size=target_size_5, mode='trilinear', align_corners=False)
        cat5 = torch.cat([feat5, blend11_up], dim=1)
        blend5 = self.blend5(cat5)
        
        if gamma is not None and beta is not None:
            B_f = gamma.shape[0]
            blend5 = blend5 * (1 + gamma.view(B_f, -1, 1, 1, 1)) + beta.view(B_f, -1, 1, 1, 1)
        
        B, C, T, H, W = blend5.shape
        final_feat = F.interpolate(blend5, size=(T, 224, 224), mode='trilinear', align_corners=False)
        final_feat_2d = final_feat.transpose(1, 2).reshape(B*T, C, 224, 224)
        
        out = self.final_head(final_feat_2d) 
        out = out + self.center_bias()
        out = self.sigmoid(out)
        
        out = out.view(B, T, 1, 224, 224).transpose(1, 2)
        return out
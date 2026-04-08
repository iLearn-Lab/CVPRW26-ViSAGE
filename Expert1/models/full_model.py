

import torch
import torch.nn as nn

from .decoder import CVMMDecoder
from .internvideo2_wrapper import InternVideo2Wrapper



class VSPModel(nn.Module):
    def __init__(self, pretrained_path, extract_layers=[11, 23, 35, 47], embed_dim=256, freeze_backbone=True):
        super().__init__()
        

        self.encoder = InternVideo2Wrapper(
            pretrained_path=pretrained_path,
            num_frames=20, 
            extract_layers=extract_layers,
            freeze_backbone=freeze_backbone
        )
        
        
        self.encoder.bfloat16()
        
        for name, param in self.encoder.named_parameters():
            if 'lora' in name:
                param.data = param.data.float()


        self.dim_reducers = nn.ModuleList([
            nn.Conv3d(in_channels=3200, out_channels=1024, kernel_size=1) 
            for _ in range(len(extract_layers))
        ])
        

        self.decoder = CVMMDecoder(
            channel_list=[1024, 1024, 1024, 1024], 
            embed_dim=embed_dim
        )
        self.dim_reducers.float()
        self.decoder.float()
        self.film_gamma = nn.Linear(3200, embed_dim)
        self.film_beta = nn.Linear(3200, embed_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x):
        features, cls_token = self.encoder(x.to(torch.bfloat16))
        cls_token = cls_token.float()
        
        gamma = self.film_gamma(cls_token)  # (B, 256)
        beta = self.film_beta(cls_token)   
        
        fp32_features = []
        for i, feat in enumerate(features):
            feat_fp32 = feat.float() 
            reduced_feat = self.dim_reducers[i](feat_fp32)
            fp32_features.append(reduced_feat)
            
        out = self.decoder(fp32_features, gamma=gamma, beta=beta)
        return out
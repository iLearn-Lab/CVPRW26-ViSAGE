

import torch
import torch.nn as nn


from .internvideo2_wrapper import InternVideo2Wrapper
from .decoder import CVMMDecoder


class VSPModel(nn.Module):
    def __init__(self, pretrained_path, extract_layers=(11, 23, 35, 47), embed_dim=256, freeze_backbone=True):
        super().__init__()

        self.encoder = InternVideo2Wrapper(
            pretrained_path=pretrained_path,
            num_frames=20,
            extract_layers=extract_layers,
            freeze_backbone=freeze_backbone,
        )

        self.encoder.bfloat16()
        for name, param in self.encoder.named_parameters():
            if "lora" in name:
                param.data = param.data.float()

        self.dim_reducers = nn.ModuleList(
            [nn.Conv3d(3200, 1024, kernel_size=1) for _ in range(len(extract_layers))]
        )
        self.decoder = CVMMDecoder(
            channel_list=[1024, 1024, 1024, 1024],
            embed_dim=embed_dim,
        )

        self.dim_reducers.float()
        self.decoder.float()

    def forward(self, x, return_aux=False):
        features = self.encoder(x.to(torch.bfloat16))

        reduced_features = []
        for i, feat in enumerate(features):
            reduced_features.append(self.dim_reducers[i](feat.float()))

        if return_aux:
            return self.decoder(reduced_features, return_aux=True)
        return self.decoder(reduced_features, return_aux=False)

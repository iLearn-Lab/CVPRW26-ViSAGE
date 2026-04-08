import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels, max_groups=32):
    for groups in (max_groups, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class TemporalGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = x.mean(dim=(-1, -2))  # [B, C, T]
        gate = self.depthwise(pooled)
        gate = self.pointwise(gate)
        return self.sigmoid(gate).unsqueeze(-1).unsqueeze(-1)


class ProjectionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualFusionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_group_norm(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                make_group_norm(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return self.act(out)


class SaliencyHead3D(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.pre_2d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
        )
        self.head_2d = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, output_hw=(224, 224)):
        bsz, _, steps, _, _ = x.shape
        x = self.pre_2d(x)
        x = F.interpolate(
            x,
            size=(steps, output_hw[0], output_hw[1]),
            mode="trilinear",
            align_corners=False,
        )
        x = x.transpose(1, 2).reshape(bsz * steps, x.size(1), output_hw[0], output_hw[1])
        x = self.head_2d(x)
        x = self.sigmoid(x)
        x = x.view(bsz, steps, 1, output_hw[0], output_hw[1]).transpose(1, 2)
        return x


class CVMMDecoder(nn.Module):
    def __init__(self, channel_list=(1024, 1024, 1024, 1024), embed_dim=256, output_hw=(224, 224)):
        super().__init__()
        c5, c11, c17, c23 = channel_list
        head_hidden = max(embed_dim // 2, 64)

        self.output_hw = output_hw

        self.proj5 = ProjectionBlock3D(c5, embed_dim)
        self.proj11 = ProjectionBlock3D(c11, embed_dim)
        self.proj17 = ProjectionBlock3D(c17, embed_dim)
        self.proj23 = ProjectionBlock3D(c23, embed_dim)

        self.temporal_gate = TemporalGate(embed_dim)

        self.refine23 = ResidualFusionBlock3D(embed_dim, embed_dim)
        self.fuse17 = ResidualFusionBlock3D(embed_dim * 2, embed_dim)
        self.fuse11 = ResidualFusionBlock3D(embed_dim * 2, embed_dim)
        self.fuse5 = ResidualFusionBlock3D(embed_dim * 2, embed_dim)
        self.refine3d = nn.Sequential(
            ResidualFusionBlock3D(embed_dim, embed_dim),
            ResidualFusionBlock3D(embed_dim, embed_dim),
        )

        self.main_head = SaliencyHead3D(embed_dim, head_hidden)
        self.aux_heads = nn.ModuleDict(
            {
                "p23": SaliencyHead3D(embed_dim, head_hidden),
                "p17": SaliencyHead3D(embed_dim, head_hidden),
                "p11": SaliencyHead3D(embed_dim, head_hidden),
            }
        )

    def _match_size(self, x, target_size):
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return x

    def forward(self, features, return_aux=False):
        f5, f11, f17, f23 = features

        p23 = self.proj23(f23)
        p23 = self.refine23(p23 * self.temporal_gate(p23))

        p17 = self.proj17(f17)
        p17 = self.fuse17(torch.cat([p17, self._match_size(p23, p17.shape[2:])], dim=1))

        p11 = self.proj11(f11)
        p11 = self.fuse11(torch.cat([p11, self._match_size(p17, p11.shape[2:])], dim=1))

        p5 = self.proj5(f5)
        p5 = self.fuse5(torch.cat([p5, self._match_size(p11, p5.shape[2:])], dim=1))

        refined = self.refine3d(p5)
        main_pred = self.main_head(refined, output_hw=self.output_hw)

        if not return_aux:
            return main_pred

        aux_preds = {
            "p23": self.aux_heads["p23"](p23, output_hw=self.output_hw),
            "p17": self.aux_heads["p17"](p17, output_hw=self.output_hw),
            "p11": self.aux_heads["p11"](p11, output_hw=self.output_hw),
        }
        return main_pred, aux_preds

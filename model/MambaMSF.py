import math
import torch
from torch import nn
from mamba_ssm import Mamba

class SpectralMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(SpectralMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            return torch.cat([x, pad_features], dim=1)
        return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad).permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        return x + x_proj if self.use_residual else x_proj

class SpatialMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4):
        super(SpatialMamba, self).__init__()
        self.use_residual = use_residual
        
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_recon = self.proj(x_recon)
        return x_recon + x if self.use_residual else x_recon

class MultiScaleMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4,
                 use_spe=True, use_spa=True, use_scale_4x4=True, use_scale_8x8=False):
        super(MultiScaleMamba, self).__init__()
        self.channels = channels
        self.use_residual = use_residual
        
        self.use_spe = use_spe
        self.use_spa = use_spa
        self.use_scale_4x4 = use_scale_4x4
        self.use_scale_8x8 = use_scale_8x8
        self.active_branches = sum([use_spe, use_spa, use_scale_4x4, use_scale_8x8])
        
        if use_spe:
            self.spe_branch = SpectralMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)
        if use_spa:
            self.spa_branch = SpatialMamba(channels, use_residual=use_residual, group_num=group_num)
        if use_scale_4x4:
            self.scale_branch_4x4 = nn.Sequential(
                nn.AvgPool2d(4),
                SpatialMamba(channels, use_residual=use_residual, group_num=group_num)
            )
        if use_scale_8x8:
            self.scale_branch_8x8 = nn.Sequential(
                nn.AvgPool2d(8),
                SpatialMamba(channels, use_residual=use_residual, group_num=group_num)
            )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * self.active_branches, channels * self.active_branches, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        features_list = []
        
        if self.use_spe:
            features_list.append(self.spe_branch(x))
        if self.use_spa:
            features_list.append(self.spa_branch(x))
        if self.use_scale_4x4:
            scale_x_4x4 = self.scale_branch_4x4(x)
            scale_x_4x4 = nn.functional.interpolate(scale_x_4x4, size=(x.shape[2], x.shape[3]), 
                                                  mode='bilinear', align_corners=True)
            features_list.append(scale_x_4x4)
        if self.use_scale_8x8:
            scale_x_8x8 = self.scale_branch_8x8(x)
            scale_x_8x8 = nn.functional.interpolate(scale_x_8x8, size=(x.shape[2], x.shape[3]), 
                                                  mode='bilinear', align_corners=True)
            features_list.append(scale_x_8x8)

        features = torch.cat(features_list, dim=1)
        attention_weights = self.attention(features)
        
        start_idx = 0
        fused = 0
        for i, feature in enumerate(features_list):
            end_idx = start_idx + self.channels
            fused += feature * attention_weights[:, start_idx:end_idx]
            start_idx = end_idx

        out = self.proj(fused)
        return out + x if self.use_residual else out

class MambaMSF(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, num_classes=10, 
                 use_residual=True, token_num=4, group_num=4):
        super(MambaMSF, self).__init__()
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        
        self.multi_scale = nn.Sequential(
            MultiScaleMamba(hidden_dim, token_num, use_residual, group_num),
            nn.AvgPool2d(2, 2),
            MultiScaleMamba(hidden_dim, token_num, use_residual, group_num),
            nn.AvgPool2d(2, 2),
            MultiScaleMamba(hidden_dim, token_num, use_residual, group_num)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.multi_scale(x)
        return self.cls_head(x)

# test
#if __name__ == '__main__':
#    batch, in_channels, h, w = 2, 128, 64, 64
#    x = torch.randn(batch, in_channels, h, w).to("cuda")
#    model = MambaMSF(in_channels=in_channels, hidden_dim=64, num_classes=10).to("cuda")
#    y = model(x)
#    print(f"Input shape: {x.shape}")
#    print(f"Output shape: {y.shape}")
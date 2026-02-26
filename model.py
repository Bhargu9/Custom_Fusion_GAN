# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__(); self.res_scale = res_scale
        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity: layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)
        self.b1 = block(filters * 1, False)
        self.b2 = block(filters * 2)
        self.b3 = block(filters * 3)
        self.b4 = block(filters * 4)
        self.b5 = nn.Conv2d(filters * 5, filters, 1, 1, 0, bias=True)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters))
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class SEAttention(nn.Module):
    """ Squeeze-and-Excitation (SE) block for channel attention. """
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Multi-Head Spatial Attention Block 

class SpatialMHSA(nn.Module):
    """ A fully-convolutional Multi-Head Self-Attention block. """
    def __init__(self, in_channels, num_heads=4):
        super(SpatialMHSA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        
        Q = self.query_conv(x).view(N, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = self.key_conv(x).view(N, self.num_heads, self.head_dim, H * W)
        V = self.value_conv(x).view(N, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        attention_map = torch.matmul(Q, K) / (self.head_dim ** 0.5) 
        attention_map = self.softmax(attention_map)
        
        out = torch.matmul(attention_map, V)
        
        out = out.permute(0, 1, 3, 2).contiguous().view(N, C, H, W)
        out = self.out_conv(out)
        
        return self.gamma * out + x

# Upsampling Block 

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

# Multi-Scale Generator 

class MultiScaleFusionGenerator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, n_rrdb_blocks=16, num_heads=8):
        super(MultiScaleFusionGenerator, self).__init__()
        
        self.head_10m = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.head_20m = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.head_30m = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        rrdb_body = [ResidualInResidualDenseBlock(num_features) for _ in range(n_rrdb_blocks)]
        self.RRDB_body_10m = nn.Sequential(*rrdb_body)
        self.RRDB_body_20m = nn.Sequential(*rrdb_body)
        self.RRDB_body_30m = nn.Sequential(*rrdb_body)
        
        self.conv_after_rrdb_10m = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_after_rrdb_20m = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_after_rrdb_30m = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.downsample_10m_x05 = nn.Sequential(nn.Conv2d(num_features, num_features, 3, 2, 1), nn.PReLU())
        self.upsample_30m_x2 = UpsampleBlock(num_features, scale_factor=2)

        self.pre_fusion_spatial_attn = SpatialMHSA(num_features, num_heads=num_heads)
        
        self.pre_fusion_channel_attn = SEAttention(num_features, reduction=16)

        fusion_in_channels = num_features * 3
        self.fusion_conv_1x1 = nn.Conv2d(fusion_in_channels, num_features, 1, 1, 0)
        self.mhsa_block = SpatialMHSA(num_features, num_heads=num_heads)
        self.fusion_conv_3x3 = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.final_upsample = nn.Sequential(
            UpsampleBlock(num_features, scale_factor=2), 
            UpsampleBlock(num_features, scale_factor=2), 
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_features, in_channels, 3, 1, 1),
            nn.Tanh() 
        )

    def forward(self, x10, x20, x30):
        h10 = self.head_10m(x10)
        h20 = self.head_20m(x20)
        h30 = self.head_30m(x30)

        f10 = self.RRDB_body_10m(h10) + h10
        f20 = self.RRDB_body_20m(h20) + h20
        f30 = self.RRDB_body_30m(h30) + h30
        
        f10 = self.conv_after_rrdb_10m(f10) # (h//2)
        f20 = self.conv_after_rrdb_20m(f20) # (h//4)
        f30 = self.conv_after_rrdb_30m(f30) # (h//8)

        f10_down = self.downsample_10m_x05(f10) # (h//4)
        f30_up = self.upsample_30m_x2(f30)       # (h//4)
        
        f10_attended = self.pre_fusion_spatial_attn(f10_down)
        f30_attended = self.pre_fusion_channel_attn(f30_up)
        
        f_cat = torch.cat([f10_attended, f20, f30_attended], dim=1)
        
        f_fused = self.fusion_conv_1x1(f_cat)
        f_attended = self.mhsa_block(f_fused)
        f_refined = self.fusion_conv_3x3(f_attended)

        out = self.final_upsample(f_refined)
        
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, _, _ = self.input_shape
        def discriminator_block(in_filters, out_filters):
            return [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 1, 1)), nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv2d(out_filters, out_filters, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        layers, in_filters = [], in_channels
        for _, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
        self.features = nn.Sequential(*layers)
        in_size = self.features(torch.empty(1, *self.input_shape)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    def forward(self, img):
        out = self.features(img)
        out = out.view(out.shape[0], -1)
        return self.classifier(out)

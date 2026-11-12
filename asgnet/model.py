import torch
from torch import nn
from torch.nn import functional as F


class HighPassDW3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pad = nn.ReplicationPad3d(1)
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=0, groups=channels, bias=False)

        with torch.no_grad():
            lap = torch.full((3, 3, 3), -1.0, dtype=torch.float32)
            lap[1, 1, 1] = 26.0
            self.dw.weight.zero_()
            self.dw.weight.copy_(lap.view(1, 1, 3, 3, 3).expand(channels, 1, 3, 3, 3))

        for p in self.dw.parameters():
            p.requires_grad = False

        self.postnorm = nn.InstanceNorm3d(channels, affine=False)

    def forward(self, x):
        x_pad = self.pad(x)
        y = self.dw(x_pad)
        return self.postnorm(y)


class SRU3d(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        if oup_channels % group_num != 0:
            group_num = 1
            while oup_channels % group_num != 0:
                group_num += 1

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / (torch.sum(self.gn.weight) + 1e-6)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)

        info_mask = (reweights >= self.gate_treshold).to(dtype=x.dtype)
        x_1 = info_mask * x
        x_2 = x - x_1
        return self.reconstruct(x_1, x_2)

    def reconstruct(self, x_1, x_2):
        c = x_1.size(1)
        if c % 2 != 0:
            return x_1 + x_2
        x_11, x_12 = torch.chunk(x_1, 2, dim=1)
        x_21, x_22 = torch.chunk(x_2, 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU3d(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel

        squeeze_up = max(1, self.up_channel // squeeze_radio)
        squeeze_low = max(1, self.low_channel // squeeze_radio)

        self.squeeze1 = nn.Conv3d(self.up_channel, squeeze_up, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(self.low_channel, squeeze_low, kernel_size=1, bias=False)

        self.GWC = nn.Conv3d(squeeze_up, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv3d(squeeze_up, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv3d(squeeze_low, op_channel - squeeze_low, kernel_size=1, bias=False)

        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out

        if out.size(1) % 2 != 0:
            return out
        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class ScConv3d(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5,
                 alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.SRU = SRU3d(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU3d(op_channel, alpha=alpha, squeeze_radio=squeeze_radio,
                         group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class SCConvUNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU()
        )
        self.sc_conv = ScConv3d(dim, group_num=min(dim, 4))
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.initial_conv(x)
        out = self.sc_conv(out)
        out += residual
        return self.activation(out)


class Head_feature(nn.Module):
    def __init__(self, in_chan, mid_chan=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chan, mid_chan, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid_chan),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv3d(mid_chan, 1, 1, bias=False)

    def forward(self, x):
        return self.out(self.conv(x))


class ASG_Block_SC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.context_path = nn.Sequential(
            nn.Conv3d(dim, dim, 7, 1, 3, groups=dim, bias=False),
            nn.InstanceNorm3d(dim)
        )
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, max(1, dim // 4), 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(max(1, dim // 4), 2 * dim, 1, bias=False),
        )
        self.high_freq_expert = nn.Sequential(
            HighPassDW3d(dim),
            SCConvUNetBlock(dim)
        )
        self.low_freq_expert = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            SCConvUNetBlock(dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Conv3d(dim, int(dim * 4.), 1, bias=False),
            nn.GELU(),
            nn.Conv3d(int(dim * 4.), dim, 1, bias=False)
        )
        self.norm = nn.InstanceNorm3d(dim, affine=True)

        self.divergence_term = None

    def forward(self, x):
        shortcut = x
        context_features = self.context_path(x)
        gates = self.gate_generator(context_features)
        gates = gates.reshape(x.shape[0], 2, x.shape[1], 1, 1, 1).softmax(dim=1)

        high_freq_out = self.high_freq_expert(x)
        low_freq_out = self.low_freq_expert(x)

        dynamic_features = gates[:, 0] * high_freq_out + gates[:, 1] * low_freq_out

        if self.training:
            hf_gap = high_freq_out.mean(dim=(2, 3, 4)).float()
            lf_gap = low_freq_out.mean(dim=(2, 3, 4)).float()
            self.divergence_term = F.cosine_similarity(hf_gap, lf_gap, dim=1).mean()

        return shortcut + self.final_mlp(self.norm(dynamic_features))


class SGF_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_dim * 2, in_dim, 1, bias=False),
            nn.InstanceNorm3d(in_dim),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g_aligned = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        gate = self.gate_conv(torch.cat([x, g_aligned], dim=1))
        return x * gate, g_aligned


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 2, 2, bias=False)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.proj(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))


class ASGNet_SC(nn.Module):
    """3D ASG-Net. Returns segmentation logits alone in evaluation mode."""

    def __init__(self, in_chans=1, num_classes=2, dims=(32, 64, 128, 256), depths=(2, 2, 2, 3)):
        super().__init__()
        c1, c2, c3, c4 = dims
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, c1, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.GELU()
        )

        def build_layer(dim, depth):
            return nn.Sequential(*[ASG_Block_SC(dim) for _ in range(depth)])

        self.encoder1 = build_layer(c1, depths[0])
        self.down1 = Downsample(c1, c2)
        self.encoder2 = build_layer(c2, depths[1])
        self.down2 = Downsample(c2, c3)
        self.encoder3 = build_layer(c3, depths[2])
        self.down3 = Downsample(c3, c4)
        self.encoder4 = build_layer(c4, depths[3])

        self.aux1, self.aux2, self.aux3 = Head_feature(c1), Head_feature(c2), Head_feature(c3)

        self.bottleneck = nn.Sequential(SCConvUNetBlock(c4), SCConvUNetBlock(c4))

        self.up3, self.sgf3 = Upsample(c4, c3), SGF_Module(c3)
        self.conv3 = nn.Sequential(nn.Conv3d(c3 * 2, c3, 3, 1, 1, bias=False), nn.InstanceNorm3d(c3), nn.GELU())
        self.decoder3 = build_layer(c3, depths[2])

        self.up2, self.sgf2 = Upsample(c3, c2), SGF_Module(c2)
        self.conv2 = nn.Sequential(nn.Conv3d(c2 * 2, c2, 3, 1, 1, bias=False), nn.InstanceNorm3d(c2), nn.GELU())
        self.decoder2 = build_layer(c2, depths[1])

        self.up1, self.sgf1 = Upsample(c2, c1), SGF_Module(c1)
        self.conv1 = nn.Sequential(nn.Conv3d(c1 * 2, c1, 3, 1, 1, bias=False), nn.InstanceNorm3d(c1), nn.GELU())
        self.decoder1 = build_layer(c1, depths[0])

        self.seg_head = nn.Conv3d(c1, num_classes, 1)

        self.asg_blocks = []
        for m in self.modules():
            if isinstance(m, ASG_Block_SC):
                self.asg_blocks.append(m)

    def forward(self, x):
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.encoder4(self.down3(x3))

        z = self.bottleneck(x4)

        x3_f, g3_a = self.sgf3(x=x3, g=self.up3(z))
        z = self.decoder3(self.conv3(torch.cat([x3_f, g3_a], 1)))

        x2_f, g2_a = self.sgf2(x=x2, g=self.up2(z))
        z = self.decoder2(self.conv2(torch.cat([x2_f, g2_a], 1)))

        x1_f, g1_a = self.sgf1(x=x1, g=self.up1(z))
        features = self.decoder1(self.conv1(torch.cat([x1_f, g1_a], 1)))

        out_main = self.seg_head(features)
        if self.training:
            diversity = torch.stack([block.divergence_term for block in self.asg_blocks]).mean()
            for block in self.asg_blocks:
                block.divergence_term = None
            return out_main, features, (self.aux1(x1), self.aux2(x2), self.aux3(x3)), diversity
        return out_main

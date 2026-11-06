import torch
from torch import nn
from torch.nn import functional as F
def resize_bilinear(x, size=None, scale_factor=None):
    """PyTorch 2.0 lacks CUDA bf16 bilinear upsampling; compute only this op in fp32."""
    dtype = x.dtype
    output = F.interpolate(x.float(), size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return output.to(dtype)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_channels, out_channels, 1, dilation=1) if rate == 1
            else ConvBNReLU(in_channels, out_channels, 3, dilation=rate)
            for rate in rates
        ])
        self.project = ConvBNReLU(len(rates) * out_channels, out_channels, 1)

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))

class TextureEncoder(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, channels, stride=2), ConvBNReLU(channels, channels))
        self.context = ASPP(channels, channels)
        self.down = nn.Sequential(nn.AvgPool2d(2), ASPP(channels, channels))

    def forward(self, x):
        return self.down(self.context(self.stem(x)))

class MaterialAwareAttention(nn.Module):
    """Learnable, Gabor-inspired local texture enhancement plus SE recalibration."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.texture_enhance = ConvBNReLU(channels, channels)
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.texture_enhance(x)
        return x * self.channel_gate(x)

class ASPPFPNFusion(nn.Module):
    def __init__(self, channels=(96, 192, 384, 768), out_channels=96):
        super().__init__()
        self.deep_aspp = ASPP(channels[3], out_channels)
        self.lateral = nn.ModuleList([nn.Conv2d(channel, out_channels, 1) for channel in channels[:3]])
        self.refine = nn.ModuleList([ConvBNReLU(out_channels, out_channels) for _ in range(3)])

    def forward(self, s2, s3, s4, s5, texture):
        p5 = self.deep_aspp(s5)
        p4 = self.refine[2](resize_bilinear(p5, size=s4.shape[-2:]) + self.lateral[2](s4))
        p3 = self.refine[1](resize_bilinear(p4, size=s3.shape[-2:]) + self.lateral[1](s3))
        f2 = self.lateral[0](s2) + texture
        return self.refine[0](resize_bilinear(p3, size=f2.shape[-2:]) + f2)

class EdgeEnhancedDecoder(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.refine = ConvBNReLU(channels, channels)
        self.decode = ConvBNReLU(channels, channels // 2)
        self.segmentation = nn.Conv2d(channels // 2, num_classes, 1)
        self.edge = nn.Conv2d(channels, 1, 1)

    def forward(self, x, output_size):
        refined = self.refine(x)
        edge = self.edge(refined)
        logits = self.segmentation(self.decode(refined))
        logits = resize_bilinear(logits, size=output_size)
        return logits, edge

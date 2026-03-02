import math
import torch.nn.functional as F
from model.PGDA_block import PGDA
from model.PMWA_block import PMWA
import torch
import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self, ch, act=nn.GELU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=bias)
        self.act = act()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=bias)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y

class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch, act=nn.GELU, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=bias)
        self.act = act()
        self.rb = ResBlock(out_ch, act=act, bias=bias)

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.rb(x)
        return x

class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, act=nn.GELU, bias=True, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)
        self.act = act()
        self.rb = ResBlock(out_ch, act=act, bias=bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode, align_corners=False if self.mode in ('bilinear', 'bicubic') else None)
        x = self.act(self.conv(x))
        x = self.rb(x)
        return x

def _pad_to_multiple(x, multiple: int, mode='reflect'):
    B, C, H, W = x.shape
    newH = int(math.ceil(H / multiple) * multiple)
    newW = int(math.ceil(W / multiple) * multiple)
    pad_h = newH - H
    pad_w = newW - W
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    if pad_h == 0 and pad_w == 0:
        return (x, (0, 0, 0, 0))
    x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    return (x, (pad_l, pad_r, pad_t, pad_b))

def _crop_from_pad(x, pads):
    pad_l, pad_r, pad_t, pad_b = pads
    if pad_l == pad_r == pad_t == pad_b == 0:
        return x
    H, W = (x.shape[-2], x.shape[-1])
    return x[..., pad_t:H - pad_b, pad_l:W - pad_r]

class UNet_PMWA_PGDA(nn.Module):

    def __init__(self, c0=16, c1=32, c2=64, c3=128, c_psf=30, bayer=True, pmwa_k=5, window_size=8, heads_s0=8, heads_s1=8, heads_s2=8, m_s0=8, m_s1=8, m_s2=4, input_resolution_hint=(256, 256), use_pmwa_out=True, heads_out=4, m_out=8, pgda_heads=8, pgda_k=10, pgda_reduction=16, dec_resblocks=2, res_scale=1, pad_multiple=64):
        super().__init__()
        assert c0 == 16, 'Default assumes stem produces 16 channels; adjust carefully if changing.'
        assert c1 % heads_s0 == 0 and c2 % heads_s1 == 0 and (c3 % heads_s2 == 0)
        assert c0 % heads_out == 0, 'heads_out must divide 16 (e.g., 4 or 8).'
        assert c3 % pgda_heads == 0
        self.c_psf = c_psf
        self.window_size = window_size
        self.pad_multiple = int(pad_multiple)
        self.res_scale = float(res_scale)
        self.bayer = bool(bayer)
        self.c_in = 3 if self.bayer else 1
        self.c_out = self.c_in
        self.stem = nn.Sequential(nn.Conv2d(self.c_in, c0, 3, 1, 1, bias=True), nn.GELU(), ResBlock(c0, act=nn.GELU))
        self.psf_down1 = nn.AvgPool2d(2, 2)
        self.psf_down2 = nn.AvgPool2d(2, 2)
        self.psf_down3 = nn.AvgPool2d(2, 2)
        self.down0 = DownBlock(c0, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.pmwa0 = PMWA(C_psf=c_psf, C_img=c1, k=pmwa_k, m=m_s0, window_size=window_size, num_heads=heads_s0, input_resolution=(input_resolution_hint[0] // 2, input_resolution_hint[1] // 2))
        self.pmwa1 = PMWA(C_psf=c_psf, C_img=c2, k=pmwa_k, m=m_s1, window_size=window_size, num_heads=heads_s1, input_resolution=(input_resolution_hint[0] // 4, input_resolution_hint[1] // 4))
        self.pmwa2 = PMWA(C_psf=c_psf, C_img=c3, k=pmwa_k, m=m_s2, window_size=window_size, num_heads=heads_s2, input_resolution=(input_resolution_hint[0] // 8, input_resolution_hint[1] // 8))
        self.psf_proj = nn.Sequential(nn.Conv2d(c_psf, c3, 1, 1, 0, bias=True), nn.GELU())
        self.pgda1 = PGDA(channels=c3, k=pgda_k, reduction=pgda_reduction, num_heads=pgda_heads)
        self.pgda2 = PGDA(channels=c3, k=pgda_k, reduction=pgda_reduction, num_heads=pgda_heads)
        self.pgda3 = PGDA(channels=c3, k=pgda_k, reduction=pgda_reduction, num_heads=pgda_heads)
        self.up2 = UpBlock(c3, c2)
        self.fuse2 = nn.Conv2d(c2 + c2, c2, 1, 1, 0, bias=True)
        self.dec2 = nn.Sequential(*[ResBlock(c2) for _ in range(dec_resblocks)])
        self.up1 = UpBlock(c2, c1)
        self.fuse1 = nn.Conv2d(c1 + c1, c1, 1, 1, 0, bias=True)
        self.dec1 = nn.Sequential(*[ResBlock(c1) for _ in range(dec_resblocks)])
        self.up0 = UpBlock(c1, c0)
        self.fuse0 = nn.Conv2d(c0 + c0, c0, 1, 1, 0, bias=True)
        self.dec0 = nn.Sequential(*[ResBlock(c0) for _ in range(dec_resblocks)])
        self.use_pmwa_out = bool(use_pmwa_out)
        if self.use_pmwa_out:
            self.pmwa_out = PMWA(C_psf=c_psf, C_img=c0, k=pmwa_k, m=m_out, window_size=window_size, num_heads=heads_out, input_resolution=input_resolution_hint)
        else:
            self.pmwa_out = None
        self.head = nn.Sequential(nn.Conv2d(c0, c0, 3, 1, 1, bias=True), nn.GELU(), nn.Conv2d(c0, self.c_out, 3, 1, 1, bias=True))

    def forward(self, x, psf):
        assert x.dim() == 4 and psf.dim() == 4
        B, Cx, H, W = x.shape
        B2, Cp, H2, W2 = psf.shape
        assert Cx == self.c_in, f'Expected x with {self.c_in} channels, got {Cx}'
        assert Cp == self.c_psf and B == B2 and (H == H2) and (W == W2)
        x_pad, pads = _pad_to_multiple(x, self.pad_multiple, mode='reflect')
        psf_pad, _ = _pad_to_multiple(psf, self.pad_multiple, mode='reflect')
        psf0 = psf_pad
        psf1 = self.psf_down1(psf0)
        psf2 = self.psf_down2(psf1)
        psf3 = self.psf_down3(psf2)
        f0 = self.stem(x_pad)
        x1 = self.down0(f0)
        x1 = self.pmwa0(x1, psf1)
        skip1 = x1
        x2 = self.down1(x1)
        x2 = self.pmwa1(x2, psf2)
        skip2 = x2
        x3 = self.down2(x2)
        x3 = self.pmwa2(x3, psf3)
        skip3 = x3
        p3 = self.psf_proj(psf3)
        b = self.pgda1(skip3, p3)
        b = self.pgda2(b, p3)
        b = self.pgda3(b, p3)
        u2 = self.up2(b)
        u2 = self.fuse2(torch.cat([u2, skip2], dim=1))
        u2 = self.dec2(u2)
        u1 = self.up1(u2)
        u1 = self.fuse1(torch.cat([u1, skip1], dim=1))
        u1 = self.dec1(u1)
        u0 = self.up0(u1)
        u0 = self.fuse0(torch.cat([u0, f0], dim=1))
        u0 = self.dec0(u0)
        if self.pmwa_out is not None:
            u0 = self.pmwa_out(u0, psf0)
        raw = self.head(u0)
        delta = self.res_scale * torch.tanh(raw)
        y = torch.clamp(x_pad + delta, 0.0, 1.0)
        y = _crop_from_pad(y, pads)
        return y
import torch
import torch.nn as nn

def count_params(model: nn.Module, trainable_only: bool=True) -> int:
    if trainable_only:
        return sum((p.numel() for p in model.parameters() if p.requires_grad))
    return sum((p.numel() for p in model.parameters()))

@torch.no_grad()
def quick_forward_test(model: nn.Module, H: int=256, W: int=256, device: str='cpu'):
    model.eval().to(device)
    x = torch.rand(1, 3, H, W, device=device)
    psf = torch.randn(1, 30, H, W, device=device)
    y = model(x, psf)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = UNet_PMWA_PGDA(input_resolution_hint=(256, 256), window_size=8, use_pmwa_out=True, heads_out=4, m_out=8, dec_resblocks=2, res_scale=0.5, pad_multiple=64)
    params_trainable = count_params(net, trainable_only=True)
    params_all = count_params(net, trainable_only=False)
    quick_forward_test(net, H=256, W=256, device=device)
    try:
        from thop import profile
        x = torch.randn(1, 3, 256, 256).to(device)
        psf = torch.randn(1, 30, 256, 256).to(device)
        net = net.to(device).eval()
        macs, params = profile(net, inputs=(x, psf), verbose=False)
    except Exception as e:
        pass
if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class LACA(nn.Module):

    def __init__(self, channels: int, k: int=10, reduction: int=16, negative_slope: float=0.1):
        super().__init__()
        assert k >= 1
        assert reduction >= 1
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d((k, k))
        self.attn = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1, bias=True), nn.LeakyReLU(negative_slope=negative_slope, inplace=True), nn.Conv2d(hidden, channels, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.pool(x)
        att = self.attn(y)
        att = F.interpolate(att, size=(h, w), mode='bilinear', align_corners=False)
        return x * att

class DeformableConv2d(nn.Module):

    def __init__(self, in_channels_blur, in_channels_psf, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.offset_conv = nn.Conv2d(in_channels_blur + in_channels_psf, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=self.dilation, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.modulator_conv = nn.Conv2d(in_channels_blur + in_channels_psf, 1 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=self.dilation, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)
        self.regular_conv = nn.Conv2d(in_channels=in_channels_blur, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=self.dilation, bias=bias)

    def forward(self, x, psf):
        offset = self.offset_conv(torch.cat((x, psf), dim=1))
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(torch.cat((x, psf), dim=1)))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding, mask=modulator, stride=self.stride, dilation=self.dilation)
        return x

class Mlp(nn.Module):

    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MultiheadCrossAttention(nn.Module):

    def __init__(self, dim_q, dim_kv, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim_q % num_heads == 0
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, 2 * dim_q, bias=qkv_bias)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)
        self.attn_drop_p = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in, kv_in, attn_mask=None, kv_padding_mask=None, is_causal=False):
        B, Nq, Cq = q_in.shape
        B2, Nk, Ckv = kv_in.shape
        assert B == B2 and Cq == self.dim_q and (Ckv == self.dim_kv)
        q = self.q(q_in).view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv_in).view(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = (kv[0], kv[1])
        if kv_padding_mask is not None:
            kpm = kv_padding_mask[:, None, None, :]
            if attn_mask is None:
                attn_mask = kpm
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask | kpm
            else:
                neg_inf = torch.finfo(attn_mask.dtype).min
                attn_mask = attn_mask + kpm.to(attn_mask.dtype) * neg_inf
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, Nq, self.dim_q)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class CrossAtten_block(nn.Module):

    def __init__(self, dim_q, dim_kv, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.drop_path_p = float(drop_path)
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.xattn = MultiheadCrossAttention(dim_q, dim_kv, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim_q)
        self.mlp = Mlp(dim_q, hidden_dim=int(dim_q * mlp_ratio), drop=drop)

    def drop_path(self, x):
        p = self.drop_path_p
        if p == 0.0 or not self.training:
            return x
        keep = 1.0 - p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = torch.floor(rnd)
        return x.div(keep) * mask

    def forward(self, x, kv, attn_mask=None, kv_padding_mask=None, is_causal=False):
        B, Cq, Hq, Wq = x.shape
        B2, Ckv, Hk, Wk = kv.shape
        assert B == B2 and Cq == self.norm_q.normalized_shape[0] and (Ckv == self.norm_kv.normalized_shape[0])
        q_t = x.permute(0, 2, 3, 1).contiguous().view(B, Hq * Wq, Cq)
        kv_t = kv.permute(0, 2, 3, 1).contiguous().view(B, Hk * Wk, Ckv)
        attn_out = self.xattn(self.norm_q(q_t), self.norm_kv(kv_t), attn_mask=attn_mask, kv_padding_mask=kv_padding_mask, is_causal=is_causal)
        q_t = q_t + self.drop_path(attn_out)
        mlp_out = self.mlp(self.norm2(q_t))
        q_t = q_t + self.drop_path(mlp_out)
        y = q_t.view(B, Hq, Wq, Cq).permute(0, 3, 1, 2).contiguous()
        return y

class PGDA(nn.Module):

    def __init__(self, channels, k=10, reduction=16, negative_slope=0.1, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, dcn_kernel=3, dcn_stride=1, dcn_padding=1, dcn_dilation=1, dcn_bias=False):
        super().__init__()
        self.laca = LACA(channels, k=k, reduction=reduction, negative_slope=negative_slope)
        self.dcn = DeformableConv2d(in_channels_blur=channels, in_channels_psf=channels, out_channels=channels, kernel_size=dcn_kernel, stride=dcn_stride, padding=dcn_padding, dilation=dcn_dilation, bias=dcn_bias)
        self.xblk = CrossAtten_block(dim_q=channels, dim_kv=channels, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path)

    def forward(self, img, psf):
        psf_att = self.laca(psf)
        kv = self.dcn(img, psf_att)
        out = self.xblk(img, kv)
        return out
if __name__ == '__main__':
    psf = torch.randn(2, 256, 64, 64, device='cuda', dtype=torch.float16)
    img = torch.randn(2, 256, 64, 64, device='cuda', dtype=torch.float16)
    model = PGDA(channels=256, num_heads=8).cuda().half()
    y = model(img, psf)

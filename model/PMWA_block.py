import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def unfold_and_permute(tensor, kernel, stride=1, pad=-1):
    if pad < 0:
        pad = (kernel - 1) // 2
    tensor = F.pad(tensor, (pad, pad, pad, pad))
    tensor = tensor.unfold(2, kernel, stride)
    tensor = tensor.unfold(3, kernel, stride)
    N, C, H, W, _, _ = tensor.size()
    tensor = tensor.reshape(N, C, H, W, -1)
    tensor = tensor.permute(0, 2, 3, 1, 4).contiguous()
    return tensor

def weight_permute_reshape(tensor, F, S2):
    N, C, H, W = tensor.size()
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    tensor = tensor.reshape(N, H, W, F, S2)
    return tensor

def FAC(feat, filters, kernel_size, stride=1):
    N, C, H, W = feat.size()
    pad = (kernel_size - 1) // 2
    feat = unfold_and_permute(feat, kernel_size, stride, pad)
    weight = weight_permute_reshape(filters, C, kernel_size ** 2)
    output = feat * weight
    output = output.sum(-1)
    output = output.permute(0, 3, 1, 2)
    return output

def Block_FAC(feat, filters_lr, kernel_size, block_size):
    N, C, H, W = feat.shape
    m = block_size
    assert H % m == 0 and W % m == 0, f'H,W must be divisible by block_size m. Got H={H},W={W},m={m}'
    Hb, Wb = (H // m, W // m)
    assert filters_lr.shape[2] == Hb and filters_lr.shape[3] == Wb, f'filters_lr spatial must be (H//m, W//m)=({Hb},{Wb}), got {filters_lr.shape[2:]}'
    assert filters_lr.shape[1] == C * (kernel_size * kernel_size), f'filters_lr channel must be C*k^2={C * (kernel_size * kernel_size)}, got {filters_lr.shape[1]}'
    pad = (kernel_size - 1) // 2
    feat_u = unfold_and_permute(feat, kernel_size, stride=1, pad=pad)
    feat_u = feat_u.view(N, Hb, m, Wb, m, C, kernel_size * kernel_size)
    w = weight_permute_reshape(filters_lr, C, kernel_size * kernel_size)
    w = w.unsqueeze(2).unsqueeze(4)
    out = (feat_u * w).sum(-1)
    out = out.permute(0, 5, 1, 2, 3, 4).contiguous().view(N, C, H, W)
    return out

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-06, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x

class CMB(nn.Module):

    def __init__(self, C_in, C_out, k, use_norm=True, use_bias=True, eps=1e-06):
        super().__init__()
        assert isinstance(k, int) and k >= 1
        self.C_in = C_in
        self.C_out = C_out
        self.k = k
        out_ch = C_out * (k * k)
        self.norm = LayerNorm2d(C_in, eps=eps) if use_norm else nn.Identity()
        self.branch_relu = nn.Sequential(nn.Conv2d(C_in, out_ch, kernel_size=1, bias=use_bias), nn.LeakyReLU(inplace=True))
        self.branch_lin = nn.Conv2d(C_in, out_ch, kernel_size=1, bias=use_bias)

    def forward(self, x):
        z = self.norm(x)
        return self.branch_relu(z) + self.branch_lin(z)

class DualCMBFusion(nn.Module):

    def __init__(self, C_in, C_out, k, ds1=8, ds2=16, out_div=1, down_mode='bilinear', up_mode='bilinear', use_norm=True, use_bias=True):
        super().__init__()
        assert ds1 % 2 == 0 and ds2 % 2 == 0, 'ds1/ds2 must be even because we use a fixed 2x maxpool.'
        assert ds1 >= 2 and ds2 >= 2, 'ds1/ds2 must be >= 2.'
        assert isinstance(out_div, int) and out_div >= 1, 'out_div must be an integer >= 1.'
        self.C_in = C_in
        self.C_out = C_out
        self.k = k
        self.ds1 = ds1
        self.ds2 = ds2
        self.out_div = out_div
        self.down_mode = down_mode
        self.up_mode = up_mode
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cmb_1 = CMB(C_in=C_in, C_out=C_out, k=k, use_norm=use_norm, use_bias=use_bias)
        self.cmb_2 = CMB(C_in=C_in, C_out=C_out, k=k, use_norm=use_norm, use_bias=use_bias)

    def _interp(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode=self.down_mode, align_corners=False if self.down_mode in ('bilinear', 'bicubic') else None)

    def _up(self, x, size):
        return F.interpolate(x, size=size, mode=self.up_mode, align_corners=False if self.up_mode in ('bilinear', 'bicubic') else None)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.C_in, f'Expected C_in={self.C_in}, got {C}'
        assert H % self.out_div == 0 and W % self.out_div == 0, f'H and W must be divisible by out_div. Got H={H}, W={W}, out_div={self.out_div}'
        H_out, W_out = (H // self.out_div, W // self.out_div)
        s1 = 2.0 / float(self.ds1)
        s2 = 2.0 / float(self.ds2)
        x_pre_1 = self._interp(x, scale_factor=s1)
        x_ds_1 = self.pool(x_pre_1)
        y1 = self.cmb_1(x_ds_1)
        y1_up = self._up(y1, size=(H_out, W_out))
        x_pre_2 = self._interp(x, scale_factor=s2)
        x_ds_2 = self.pool(x_pre_2)
        y2 = self.cmb_2(x_ds_2)
        y2_up = self._up(y2, size=(H_out, W_out))
        return y1_up + y2_up

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowCrossAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_kv, mask=None):
        B_, N, C = x_q.shape
        q = self.q_proj(x_q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x_kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = (kv[0], kv[1])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinCrossTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.norm_q1 = norm_layer(dim)
        self.norm_kv1 = norm_layer(dim)
        self.attn = WindowCrossAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.register_buffer('attn_mask', self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        assert x_kv.shape == x_q.shape, 'x_q and x_kv must have same shape'
        assert C == self.dim, f'channel dim mismatch: got {C}, expect {self.dim}'
        assert H % self.window_size == 0 and W % self.window_size == 0, 'H and W must be divisible by window_size (pad beforehand if needed).'
        shortcut = x_q
        q = x_q.permute(0, 2, 3, 1).contiguous()
        kv = x_kv.permute(0, 2, 3, 1).contiguous()
        q = self.norm_q1(q)
        kv = self.norm_kv1(kv)
        if self.shift_size > 0:
            q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)
        kv_windows = window_partition(kv, self.window_size).view(-1, self.window_size * self.window_size, C)
        x_size = (H, W)
        if self.input_resolution == x_size:
            attn_windows = self.attn(q_windows, kv_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(q_windows, kv_windows, mask=self.calculate_mask(x_size).to(q_windows.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        out = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        out = out.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(out)
        x_ln = x.permute(0, 2, 3, 1).contiguous()
        x_ln = x_ln.view(B, H * W, C)
        x_ln = x_ln + self.drop_path(self.mlp(self.norm2(x_ln)))
        x_ln = x_ln.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_ln

class PMWA(nn.Module):

    def __init__(self, C_psf, C_img, k=5, m=8, ds1=8, ds2=16, window_size=8, shift_size=0, num_heads=4, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, down_mode='bilinear', up_mode='bilinear', use_norm=True, use_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm, input_resolution=(256, 256)):
        super().__init__()
        assert C_img % num_heads == 0
        self.C_psf, self.C_img, self.k, self.m, self.window_size = (C_psf, C_img, k, m, window_size)
        self.cmb = DualCMBFusion(C_in=C_psf, C_out=C_img, k=k, ds1=ds1, ds2=ds2, out_div=m, down_mode=down_mode, up_mode=up_mode, use_norm=use_norm, use_bias=use_bias)
        self.wca = SwinCrossTransformerBlock(dim=C_img, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, img, psf):
        B, Ci, H, W = img.shape
        B2, Cp, H2, W2 = psf.shape
        assert Ci == self.C_img and Cp == self.C_psf and ((H, W) == (H2, W2)) and (B == B2)
        assert H % self.m == 0 and W % self.m == 0
        assert H % self.window_size == 0 and W % self.window_size == 0
        psf_feat_lr = self.cmb(psf)
        psf_filter = Block_FAC(img, psf_feat_lr, kernel_size=self.k, block_size=self.m)
        return self.wca(img, psf_filter)

def count_params(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

@torch.no_grad()
def benchmark_pmwa_runtime(model: nn.Module, img: torch.Tensor, psf: torch.Tensor, warmup: int=20, iters: int=50):
    device = img.device
    model.eval()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(warmup):
            _ = model(img, psf)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(iters):
            _ = model(img, psf)
        ender.record()
        torch.cuda.synchronize()
        avg_ms = starter.elapsed_time(ender) / iters
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        return (avg_ms, peak_mem_mb)
    else:
        for _ in range(warmup):
            _ = model(img, psf)
        t0 = time.time()
        for _ in range(iters):
            _ = model(img, psf)
        t1 = time.time()
        avg_ms = (t1 - t0) * 1000.0 / iters
        return (avg_ms, float('nan'))

def try_thop_macs(model: nn.Module, img: torch.Tensor, psf: torch.Tensor):
    try:
        from thop import profile
        model.eval()
        macs, params = profile(model, inputs=(img, psf), verbose=False)
        return float(macs)
    except Exception as e:
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    B, C_psf, C_img = (2, 30, 16)
    H, W = (256, 256)
    m_list = [1, 2, 4, 8, 16]
    k = 5
    window_size = 8
    num_heads = 8
    input_resolution = (256, 256)
    psf = torch.randn(B, C_psf, H, W, device=device, dtype=dtype)
    img = torch.randn(B, C_img, H, W, device=device, dtype=dtype)
    for m in m_list:
        if H % m != 0 or W % m != 0:
            continue
        model = PMWA(C_psf=C_psf, C_img=C_img, k=k, m=m, ds1=8, ds2=16, window_size=window_size, shift_size=0, num_heads=num_heads, input_resolution=input_resolution).to(device=device, dtype=dtype)
        params_m = count_params(model) / 1000000.0
        macs = try_thop_macs(model, img, psf)
        macs_g = macs / 1000000000.0 if macs is not None else float('nan')
        avg_ms, peak_mem_mb = benchmark_pmwa_runtime(model, img, psf, warmup=20, iters=50)
if __name__ == '__main__':
    main()

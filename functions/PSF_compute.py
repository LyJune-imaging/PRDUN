import math
import numpy as np
import torch
import torch.nn.functional as F
device = 'cuda'

def img2polar(img, numRadii=None, center=None, border='constant', a_sampling=4):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    C, H, W = img.shape
    maxSize = max(img.shape)
    if numRadii is None:
        numRadii = maxSize
    initialAngle = math.pi / 4
    finalAngle = 2 * math.pi + math.pi / 4
    if maxSize > 700:
        numAngles = int(a_sampling * maxSize * ((finalAngle - initialAngle) / (2 * math.pi)))
    else:
        numAngles = int(3 * maxSize * ((finalAngle - initialAngle) / (2 * math.pi)))
    step = H / 2.0 / numRadii
    lin = torch.arange(0, numRadii, device=img.device, dtype=img.dtype) * step
    radii = math.sqrt(2.0) * (lin + 0.5)
    ang_lin = torch.linspace(initialAngle, finalAngle, steps=numAngles + 1, device=img.device, dtype=img.dtype)[:-1]
    r = radii.unsqueeze(0).expand(numAngles, numRadii)
    theta = ang_lin.unsqueeze(1).expand(numAngles, numRadii)
    if center is None:
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
    else:
        cx, cy = center
    xCart = r * torch.cos(theta) + cx
    yCart = r * torch.sin(theta) + cy
    pad = 3
    if border == 'constant':
        img_pad = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode='replicate')[0]
        xCart = xCart + pad
        yCart = yCart + pad
        H2, W2 = (H + 2 * pad, W + 2 * pad)
    else:
        img_pad = img
        H2, W2 = (H, W)
    gx = 2.0 * (xCart / (W2 - 1)) - 1.0
    gy = 2.0 * (yCart / (H2 - 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    polarImage = F.grid_sample(img_pad.unsqueeze(0), grid, mode='bicubic', padding_mode='zeros', align_corners=True)
    return polarImage.squeeze(0)

def smoothstep(x):
    return x * x * (3 - 2 * x)

def circ(r, radius, eps=0.005, diff=True):
    if diff:
        t = (radius - r) / eps
        t = torch.clamp(t, 0.0, 1.0)
        return smoothstep(t)
    else:
        return (r.abs() <= radius).to(r.dtype)

def shift_torch(img: torch.Tensor, shift: tuple, mode: str='bilinear') -> torch.Tensor:
    if img.ndim == 2:
        img_in = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img_in = img.unsqueeze(0)
    else:
        img_in = img.unsqueeze(0)
    B, C, H, W = img_in.shape
    ys = torch.linspace(0, H - 1, H, device=img.device)
    xs = torch.linspace(0, W - 1, W, device=img.device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
    dy, dx = shift
    x_shift = x_grid - dx
    y_shift = y_grid - dy
    x_norm = 2.0 * x_shift / (W - 1) - 1.0
    y_norm = 2.0 * y_shift / (H - 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
    shifted = F.grid_sample(img_in, grid, mode=mode, padding_mode='zeros', align_corners=True)
    out = shifted.squeeze(0)
    if img.ndim == 2:
        return out.squeeze(0)
    return out

def compute_pupil_phase(coeffs, X, Y, u, v, higher_order=None):
    rot_angle = torch.atan2(v, u)
    obj_rad = torch.sqrt(u ** 2 + v ** 2)
    X_rot = X * torch.cos(rot_angle) + Y * torch.sin(rot_angle)
    Y_rot = -X * torch.sin(rot_angle) + Y * torch.cos(rot_angle)
    pupil_radii = torch.square(X_rot) + torch.square(Y_rot)
    pupil_phase = coeffs[0] * torch.square(pupil_radii) + coeffs[1] * obj_rad * pupil_radii * X_rot + coeffs[2] * obj_rad ** 2 * torch.square(X_rot) + coeffs[3] * obj_rad ** 2 * pupil_radii + coeffs[4] * obj_rad ** 3 * X_rot + coeffs[5] * pupil_radii
    if higher_order is not None:
        higher_order_pupil = higher_order[0] * pupil_radii ** 3 + higher_order[1] * obj_rad * pupil_radii ** 2 * X_rot + higher_order[2] * obj_rad ** 2 * pupil_radii * X_rot ** 2 + higher_order[3] * obj_rad ** 2 * pupil_radii ** 2 + higher_order[4] * obj_rad ** 3 * pupil_radii * X_rot + higher_order[5] * obj_rad ** 4 * X_rot ** 2 + higher_order[6] * obj_rad ** 4 * pupil_radii + higher_order[7] * obj_rad ** 5 * X_rot
        pupil_phase += higher_order_pupil
    return pupil_phase

def get_rdm_psfs(seidel_coeffs: torch.Tensor, dim: int, sys_params: dict={}, downsample: int=1, higher_order=None, verbose: bool=True, device: torch.device=torch.device('cpu'), diff=True, patch_size: int=1):
    if not isinstance(seidel_coeffs, torch.Tensor):
        seidel_coeffs = torch.tensor(seidel_coeffs, dtype=torch.float32, device=device)
    else:
        seidel_coeffs = seidel_coeffs.to(device)
    def_sys = {'samples': dim, 'L': 0.0, 'lamb': 5.5e-07, 'NA': 0.5}
    radius_over_z = torch.tan(torch.asin(torch.tensor(def_sys['NA'], device=device)))
    def_sys['L'] = dim * def_sys['lamb'] / (4 * radius_over_z.item())
    def_sys.update(sys_params)
    rs = torch.linspace(0.0, dim / 2, steps=dim // abs(downsample), device=device)
    point_list = [(r, -r) for r in rs]
    num_radii = len(point_list)
    if patch_size <= 0:
        raise ValueError(f'patch_size must be >=1, got {patch_size}')
    group_indices = torch.arange(num_radii, device=device) // patch_size
    unique_groups, inverse_indices = torch.unique(group_indices, return_inverse=True)
    group_centers = []
    for grp in unique_groups:
        idxs = torch.where(group_indices == grp)[0]
        mid_idx = idxs[len(idxs) // 2]
        group_centers.append(point_list[mid_idx])
    if verbose:
        pass
    psf_data = compute_rdm_psfs(seidel_coeffs, group_centers, sys_params=def_sys, polar=True, stack=True, buffer=2, downsample=downsample, higher_order=higher_order, device=device, diff=diff)
    seg = psf_data[..., :-2, :]
    fft1 = torch.fft.rfft(seg, dim=1)
    real = fft1.real
    imag = fft1.imag
    psf_data = torch.cat([real, imag], dim=1)
    return psf_data

def compute_rdm_psfs(coeffs, desired_list, dim=None, sys_params={}, polar=False, stack=False, buffer=2, shift=True, downsample=1, higher_order=None, device=torch.device('cuda'), diff=True):
    if dim is None and sys_params == {}:
        raise NotImplementedError
    if dim is None:
        dim = sys_params['samples']
    else:
        def_sys_params = {'samples': dim, 'L': 0, 'lamb': 5.5e-07, 'NA': 0.5}
        radius_over_z = math.tan(math.asin(def_sys_params['NA']))
        def_sys_params['L'] = dim * def_sys_params['lamb'] / (4 * radius_over_z)
        def_sys_params.update(sys_params)
        sys_params = def_sys_params
    dtype = torch.float32
    desired_pts = [(torch.as_tensor(p[0], device=device, dtype=dtype), torch.as_tensor(p[1], device=device, dtype=dtype)) for p in desired_list]
    iterable_coords = desired_pts
    samples = sys_params['samples']
    L = sys_params['L']
    dt = L / samples
    lamb = sys_params['lamb']
    radius_over_z = math.tan(math.asin(sys_params['NA']))
    k = 2 * math.pi / lamb
    fx = torch.linspace(-1 / (2 * dt), 1 / (2 * dt), samples, device=device)
    Fx, Fy = torch.meshgrid(fx, fx, indexing='xy')
    scale_factor = lamb / radius_over_z
    circle = circ(torch.sqrt(Fx ** 2 + Fy ** 2) * scale_factor, radius=1, diff=diff)
    if stack:
        if polar:
            desired_psfs = torch.zeros((len(desired_pts), int(samples // downsample) * 3 + buffer, int(samples // downsample)), device=device)
        else:
            desired_psfs = torch.zeros((len(desired_pts), int(samples // downsample), int(samples // downsample) + buffer), device=device)
    else:
        desired_psfs = []
    if higher_order is not None:
        higher_order = lamb * higher_order
    idx = 0
    for u_pt, v_pt in iterable_coords:
        W = compute_pupil_phase(lamb * coeffs, X=-Fx * scale_factor, Y=-Fy * scale_factor, u=u_pt / (samples / 2), v=-v_pt / (samples / 2), higher_order=higher_order)
        H = circle * torch.exp(-1j * k * W)
        curr = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(H)))
        curr_psf = torch.abs(curr) ** 2
        denom = curr_psf.sum().clamp_min(1e-08)
        curr_psf = curr_psf / denom
        if shift:
            curr_psf = shift_torch(curr_psf, shift=(-v_pt, u_pt), mode='bilinear')
        if downsample != 1:
            curr_psf = F.interpolate(curr_psf.unsqueeze(0).unsqueeze(0), scale_factor=1 / downsample, mode='bilinear', align_corners=False).squeeze()
        if polar:
            curr_psf = img2polar(curr_psf.float(), numRadii=int(dim // downsample))
        if stack:
            if polar:
                if buffer > 0:
                    desired_psfs[idx, :-buffer, :] = curr_psf
                else:
                    desired_psfs[idx, :, :] = curr_psf
            elif buffer > 0:
                desired_psfs[idx, :, :-buffer] = curr_psf
            else:
                desired_psfs[idx, :, :] = curr_psf
        else:
            desired_psfs.append(curr_psf)
        idx += 1
    return desired_psfs

def seidel2psfw(seidel, dim, diff, patch_size=1, high_order=None):
    psf_roft = get_rdm_psfs(seidel, dim=dim, device=device, diff=diff, patch_size=patch_size, higher_order=high_order)
    M, R = (psf_roft.shape[1], psf_roft.shape[2])
    r_list = np.sqrt(2) * (np.linspace(0, dim / 2, dim, endpoint=False) + 0.5)
    r_list = torch.tensor(r_list, dtype=torch.float32, device=device)
    dr = r_list[1] - r_list[0]
    dtheta = 2 * math.pi / M
    weights = (r_list * dr * dtheta).view(1, R, 1)
    pr0 = psf_roft[:, :M // 2, :]
    pi0 = psf_roft[:, M // 2:, :]
    psf_complex = (pr0 + 1j * pi0).permute(1, 0, 2).to(device)
    psf_weighted = (psf_complex * weights).to(torch.complex64)
    return psf_weighted

def _center_crop_hw(x: torch.Tensor, m: int) -> torch.Tensor:
    H, W = (x.shape[-2], x.shape[-1])
    if m > H or m > W:
        raise ValueError(f'crop size m={m} must be <= H,W ({H},{W})')
    y0 = (H - m) // 2
    x0 = (W - m) // 2
    return x[..., y0:y0 + m, x0:x0 + m]

def rotate_kernels_batch(kernels: torch.Tensor, angles: torch.Tensor, chunk_size: int=4096, force_no_cudnn: bool=True) -> torch.Tensor:
    B, C, kH, kW = kernels.shape
    assert C == 1 and kH == kW, 'kernels must be (B,1,k,k)'
    device = kernels.device
    dtype = kernels.dtype
    kernels = kernels.contiguous()
    angles = angles.contiguous()
    outs = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        ker = kernels[s:e].contiguous()
        ang = angles[s:e].to(dtype).contiguous()
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)
        theta = torch.zeros((e - s, 2, 3), device=device, dtype=dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, size=(e - s, 1, kH, kW), align_corners=True).contiguous()
        if force_no_cudnn and device.type == 'cuda':
            with torch.backends.cudnn.flags(enabled=False):
                rot = F.grid_sample(ker, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        else:
            rot = F.grid_sample(ker, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        outs.append(rot)
    return torch.cat(outs, dim=0)

@torch.no_grad()
def compute_rp_lut_from_full_psfs(psf_full_rgb: torch.Tensor, energy_frac: float=0.8) -> torch.Tensor:
    ef = float(energy_frac)
    ef = max(1e-06, min(1.0 - 1e-06, ef))
    N, C, H, W = psf_full_rgb.shape
    assert C == 3
    device = psf_full_rgb.device
    dtype = torch.float32
    yc = H // 2
    xc = W // 2
    y = torch.arange(H, device=device, dtype=dtype) - yc
    x = torch.arange(W, device=device, dtype=dtype) - xc
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx * xx + yy * yy).reshape(-1)
    order = torch.argsort(r)
    r_sorted = r[order]
    rp_lut = torch.empty((3, N), device=device, dtype=dtype)
    for i in range(N):
        for c in range(3):
            psf = psf_full_rgb[i, c].to(dtype)
            psf = psf / (psf.sum() + 1e-12)
            flat_idx = torch.argmax(psf.reshape(-1))
            y0 = (flat_idx // W).item()
            x0 = (flat_idx % W).item()
            dy = yc - y0
            dx = xc - x0
            psf_c = torch.roll(psf, shifts=(dy, dx), dims=(0, 1))
            flat = psf_c.reshape(-1)[order]
            cdf = torch.cumsum(flat, dim=0)
            j = torch.searchsorted(cdf, torch.tensor(ef, device=device, dtype=dtype), right=False)
            j = torch.clamp(j, 0, r_sorted.numel() - 1)
            rp_lut[c, i] = r_sorted[j]
    return rp_lut

def build_radius_angle_index_maps(H: int, W: int, N: int, angle_bins: int, theta_ref: float=-math.pi / 4, device: str='cuda', r_max: float | None=None):
    device = torch.device(device)
    dtype = torch.float32
    yc = (H - 1) / 2.0
    xc = (W - 1) / 2.0
    y = torch.arange(H, device=device, dtype=dtype) - yc
    x = torch.arange(W, device=device, dtype=dtype) - xc
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r_map = torch.sqrt(xx * xx + yy * yy)
    phi = torch.atan2(yy, xx)
    delta = phi - float(theta_ref)
    delta = torch.remainder(delta, 2.0 * math.pi)
    a = delta / (2.0 * math.pi) * angle_bins
    idx_a = torch.floor(a).to(torch.long)
    idx_a = torch.clamp(idx_a, 0, angle_bins - 1)
    if r_max is None:
        r_max = float(min(H, W) / 2.0)
    r_map_c = torch.clamp(r_map, 0.0, float(r_max))
    rs_ring = torch.linspace(0.0, float(r_max), steps=N, device=device, dtype=dtype)
    idx_r = (r_map_c[..., None] - rs_ring[None, None, :]).abs().argmin(dim=2).to(torch.long)
    eps = 1e-12
    norm = torch.sqrt(xx * xx + yy * yy) + eps
    dir_x = -xx / norm
    dir_y = -yy / norm
    dir_map = torch.stack([dir_x, dir_y], dim=0)
    return (idx_r, idx_a, dir_map)

def build_psf_feature_map(dim: int, k: int, kc: int, se_r: torch.Tensor, se_g: torch.Tensor, se_b: torch.Tensor, sys_r: dict | None=None, sys_g: dict | None=None, sys_b: dict | None=None, device: str='cuda', buffer: int=0, shift: bool=True, angle_bins: int=180, energy_frac: float=0.8, renormalize_kernel: bool=True, default_NA: float=0.5, default_lamb0: float=5.5e-07, default_lamb_r: float=6.2e-07, default_lamb_g: float=5.5e-07, default_lamb_b: float=4.6e-07) -> torch.Tensor:
    device_t = torch.device(device)
    if sys_r is None or sys_g is None or sys_b is None:
        base_sys = {'samples': dim, 'NA': default_NA}
        radius_over_z = math.tan(math.asin(base_sys['NA']))
        base_sys['L'] = dim * default_lamb0 / (4 * radius_over_z)
        if sys_r is None:
            sys_r = dict(base_sys)
            sys_r['lamb'] = default_lamb_r
        if sys_g is None:
            sys_g = dict(base_sys)
            sys_g['lamb'] = default_lamb_g
        if sys_b is None:
            sys_b = dict(base_sys)
            sys_b['lamb'] = default_lamb_b
    N = int(dim)
    rs = np.linspace(0, dim / 2, N, endpoint=False)
    point_list = [(float(r), float(-r)) for r in rs]
    psf_r_full = compute_rdm_psfs(se_r, point_list, sys_params=sys_r, polar=False, stack=True, buffer=buffer, shift=shift, downsample=1, device=device_t).float()
    psf_g_full = compute_rdm_psfs(se_g, point_list, sys_params=sys_g, polar=False, stack=True, buffer=buffer, shift=shift, downsample=1, device=device_t).float()
    psf_b_full = compute_rdm_psfs(se_b, point_list, sys_params=sys_b, polar=False, stack=True, buffer=buffer, shift=shift, downsample=1, device=device_t).float()
    if buffer > 0:
        psf_r_full = psf_r_full[..., :-buffer]
        psf_g_full = psf_g_full[..., :-buffer]
        psf_b_full = psf_b_full[..., :-buffer]
    psf_full_rgb = torch.stack([psf_r_full, psf_g_full, psf_b_full], dim=1)
    rp_lut = compute_rp_lut_from_full_psfs(psf_full_rgb, energy_frac=energy_frac)
    psf_crop = _center_crop_hw(psf_full_rgb, k)
    ker_mean = psf_crop.mean(dim=1, keepdim=True)
    angle_bins = int(angle_bins)
    if angle_bins <= 0:
        raise ValueError(f'angle_bins must be positive, got {angle_bins}')
    angles = torch.arange(angle_bins, device=device_t, dtype=torch.float32) * (2.0 * math.pi / float(angle_bins))
    A = angles.numel()
    ker_rep = ker_mean[:, None, ...].expand(N, A, 1, k, k).reshape(N * A, 1, k, k).contiguous()
    ang_rep = angles[None, :].expand(N, A).reshape(N * A).contiguous()
    ker_rot = rotate_kernels_batch(ker_rep, ang_rep)
    if kc != k:
        ker_rot_c = F.interpolate(ker_rot, size=(kc, kc), mode='bilinear', align_corners=False)
    else:
        ker_rot_c = ker_rot
    if renormalize_kernel:
        ker_rot_c = ker_rot_c / (ker_rot_c.sum(dim=(-1, -2), keepdim=True) + 1e-12)
    ker_feat = ker_rot_c.reshape(N * A, kc * kc)
    ker_table = ker_feat.view(N, A, kc * kc)
    idx_r, idx_a, dir_map = build_radius_angle_index_maps(H=dim, W=dim, N=N, angle_bins=A, theta_ref=-math.pi / 4, device=device, r_max=float(dim / 2.0))
    lin = (idx_r * A + idx_a).reshape(-1)
    table_flat = ker_table.reshape(N * A, kc * kc)
    pix_feat = table_flat[lin]
    pix_feat = pix_feat.view(dim, dim, kc * kc).permute(2, 0, 1).contiguous()
    rp_map = rp_lut[:, idx_r]
    psf_map = torch.cat([pix_feat, rp_map, dir_map.to(pix_feat.dtype)], dim=0).contiguous()
    return psf_map

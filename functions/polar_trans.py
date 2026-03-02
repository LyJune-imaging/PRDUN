import torch
import math
from functions.mosaic_func import bayer_mosaic
import torch.nn.functional as F

def img2polar_circular_batch(imgs, num_radii, num_angles=None, center=None):
    B, C, H, W = imgs.shape
    if center is None:
        cx, cy = ((W - 1) / 2.0, (H - 1) / 2.0)
    else:
        cx, cy = center
    maxr = math.sqrt(2) * min(H, W) / 2.0
    if num_radii is None:
        num_radii = min(H, W)
    if num_angles is None:
        num_angles = int(3 * max(H, W))
    radii = torch.linspace(0, maxr, num_radii, device=imgs.device)
    angles = torch.linspace(0, 2 * math.pi, num_angles, device=imgs.device)
    ang, rad = torch.meshgrid(angles, radii, indexing='ij')
    x = rad * torch.cos(ang) + cx
    y = rad * torch.sin(ang) + cy
    x_norm = x / (W - 1) * 2 - 1
    y_norm = y / (H - 1) * 2 - 1
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(imgs, grid, mode='bicubic', padding_mode='border', align_corners=True)

def polar2img_circular_batch(polars, image_size, center=None):
    B, C, A, R = polars.shape
    H, W = image_size
    if center is None:
        cx, cy = ((W - 1) / 2.0, (H - 1) / 2.0)
    else:
        cx, cy = center
    xs = torch.arange(W, device=polars.device)
    ys = torch.arange(H, device=polars.device)
    xg, yg = torch.meshgrid(xs, ys, indexing='xy')
    dx, dy = (xg - cx, yg - cy)
    r = torch.sqrt(dx ** 2 + dy ** 2)
    theta = torch.atan2(dy, dx) % (2 * math.pi)
    maxr = math.sqrt(2) * min(H, W) / 2.0
    r_norm = r / maxr * (R - 1)
    t_norm = theta / (2 * math.pi) * (A - 1)
    x_n = r_norm / (R - 1) * 2 - 1
    y_n = t_norm / (A - 1) * 2 - 1
    x_c = x_n.clamp(-1, 1)
    y_w = (y_n + 1) % 2 - 1
    grid = torch.stack((x_c, y_w), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(polars, grid, mode='bicubic', padding_mode='border', align_corners=True)

def A_color(imgs, psf_weighted, num_angles=None, center=None, bayer=False):
    imgs_p = img2polar_circular_batch(imgs, num_radii=imgs.shape[2], num_angles=num_angles, center=center)
    Xf = torch.fft.rfft(imgs_p, dim=2, norm='ortho')
    out_fft = torch.einsum('bcmr,cmrs->bcms', Xf, psf_weighted)
    polar_filtered = torch.fft.irfft(out_fft, n=imgs_p.size(2), dim=2, norm='ortho')
    out = polar2img_circular_batch(polar_filtered, (imgs.shape[2], imgs.shape[3]), center=center)
    if bayer:
        out = out
    return out

def AT_color(y, psf_weighted, num_angles=None, center=None, bayer=False):
    y_v = y.clone().requires_grad_(True)
    y_p = A_color(y_v, psf_weighted, num_angles=num_angles, center=center, bayer=bayer)
    inner = (y_p * y).sum()
    grad = torch.autograd.grad(inner, y_v, retain_graph=False, create_graph=False)[0]
    return grad

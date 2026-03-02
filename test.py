import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import csv
import math
from pathlib import Path
import torch
import numpy as np
import lpips
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_loss
from functions.dataset import DeblurDataset, GrayDeblurDataset
from functions.PSF_compute import get_rdm_psfs, build_psf_feature_map
from model.Unroll import Unrolled_net

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def atomic_write_text(path: Path, text: str, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
_lpips_fn = lpips.LPIPS(net='alex').eval()
for p in _lpips_fn.parameters():
    p.requires_grad_(False)

def charbonnier_loss(x, y, eps=0.001):
    diff = x - y
    return torch.mean(torch.sqrt(diff * diff + eps * eps))

def rec_loss_fn(pred, gt, w_char=1.0, w_ssim=0.4, w_lpips=0.1, eps=0.001):
    l_char = charbonnier_loss(pred, gt, eps=eps)
    l_ssim = 1.0 - ssim_loss(pred, gt, data_range=1.0, size_average=True)
    pred_lp = pred * 2.0 - 1.0
    gt_lp = gt * 2.0 - 1.0
    if pred_lp.shape[1] == 1:
        pred_lp = pred_lp.repeat(1, 3, 1, 1)
        gt_lp = gt_lp.repeat(1, 3, 1, 1)
    _lpips_fn.to(pred.device)
    l_lpips = _lpips_fn(pred_lp, gt_lp).mean()
    return w_char * l_char + w_ssim * l_ssim + w_lpips * l_lpips

def _seidel_to_psfw_with_sys(seidel: torch.Tensor, dim: int, sys_params: dict, device: str, diff: bool=True, patch_size: int=1, higher_order=None):
    psf_roft = get_rdm_psfs(seidel_coeffs=seidel, dim=dim, sys_params=sys_params, downsample=1, higher_order=higher_order, verbose=True, device=torch.device(device), diff=diff, patch_size=patch_size)
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

def build_psfW_from_seidel(se_r, se_g, se_b, dim, bayer, device):
    if bayer:
        sys_r = {'samples': dim, 'L': 0.0, 'lamb': 6.5e-07, 'NA': 0.5}
        sys_g = {'samples': dim, 'L': 0.0, 'lamb': 5.5e-07, 'NA': 0.5}
        sys_b = {'samples': dim, 'L': 0.0, 'lamb': 4.8e-07, 'NA': 0.5}
        sys_r['L'] = dim * sys_r['lamb'] / (4 * np.tan(np.arcsin(sys_r['NA'])))
        sys_g['L'] = dim * sys_g['lamb'] / (4 * np.tan(np.arcsin(sys_g['NA'])))
        sys_b['L'] = dim * sys_b['lamb'] / (4 * np.tan(np.arcsin(sys_b['NA'])))
        psfW_r = _seidel_to_psfw_with_sys(se_r, dim, sys_r, device=device, diff=True)
        psfW_g = _seidel_to_psfw_with_sys(se_g, dim, sys_g, device=device, diff=True)
        psfW_b = _seidel_to_psfw_with_sys(se_b, dim, sys_b, device=device, diff=True)
        psfW = torch.stack([psfW_r, psfW_g, psfW_b], dim=0)
    else:
        sys_r = {'samples': dim, 'L': 0.0, 'lamb': 6.5e-07, 'NA': 0.5}
        sys_r['L'] = dim * sys_r['lamb'] / (4 * np.tan(np.arcsin(sys_r['NA'])))
        psfW_r = _seidel_to_psfw_with_sys(se_r, dim, sys_r, device=device, diff=True)
        psfW = torch.stack([psfW_r], dim=0)
    return psfW.to(device)

def build_psf_map_from_seidel(se_r, se_g, se_b, dim, device, k=17, kc=5, angle_bins=180, energy_frac=0.8):
    psf_map = build_psf_feature_map(dim=dim, k=k, kc=kc, se_r=se_r, se_g=se_g, se_b=se_b, device=device, angle_bins=angle_bins, energy_frac=energy_frac)
    return psf_map.to(device).float()

def check_tensor(t: torch.Tensor, name='tensor'):
    with torch.no_grad():
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        is_finite = torch.isfinite(t)
        finite_count = is_finite.sum().item()
        numel = t.numel()
        if finite_count == 0:
            return
        if torch.is_complex(t):
            tr_safe = torch.nan_to_num(t.real, nan=0.0, posinf=0.0, neginf=0.0)
            ti_safe = torch.nan_to_num(t.imag, nan=0.0, posinf=0.0, neginf=0.0)
            ta_safe = torch.nan_to_num(t.abs(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

def evaluate_and_save(model, psfW, psf_map, loader, device, save_root, bayer=True):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _lpips_fn.to(device).eval()
    for p in _lpips_fn.parameters():
        p.requires_grad_(False)
    psfW = psfW.detach()
    psf_map = psf_map.detach()

    def _to_pil_uint8_cpu(t3):
        t = t3.clamp(0, 1).numpy()
        if t.shape[0] == 1:
            arr = (t[0] * 255.0).astype('uint8')
            return Image.fromarray(arr, mode='L')
        elif t.shape[0] == 3:
            arr = (t.transpose(1, 2, 0) * 255.0).astype('uint8')
            return Image.fromarray(arr, mode='RGB')
        else:
            raise ValueError(f'Unsupported channel count: {t.shape[0]} (expected 1 or 3)')
    n_samples = 0
    ssim_sum = 0.0
    mse_sum = 0.0
    psnr_sum = 0.0
    lpips_sum = 0.0
    sample_idx = 0
    per_sample_rows = []
    for i, (gt, y) in enumerate(loader):
        gt = gt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.enable_grad():
            inp = y.detach().requires_grad_(True)
            rec_output = model(inp, psfW, psf_map)
            rec = rec_output[-1]
            del rec_output
        with torch.no_grad():
            rec_eval = rec.detach()
            gt_eval = gt.detach()
            inp_eval = inp.detach()
            B = rec_eval.shape[0]
            ssim_vals = []
            for b in range(B):
                ssim_b = ssim_loss(rec_eval[b:b + 1], gt_eval[b:b + 1], data_range=1.0, size_average=True).item()
                ssim_vals.append(ssim_b)
            mse_per = torch.mean((rec_eval - gt_eval) ** 2, dim=(1, 2, 3))
            psnr_per = 10.0 * torch.log10(1.0 / torch.clamp(mse_per, min=1e-10))
            rec_lp = rec_eval.clamp(0, 1) * 2.0 - 1.0
            gt_lp = gt_eval.clamp(0, 1) * 2.0 - 1.0
            if rec_lp.shape[1] == 1:
                rec_lp = rec_lp.repeat(1, 3, 1, 1)
                gt_lp = gt_lp.repeat(1, 3, 1, 1)
            lpips_per = _lpips_fn(rec_lp, gt_lp).view(-1)
            rec_cpu = rec_eval.float().cpu()
            gt_cpu = gt_eval.float().cpu()
            inp_cpu = inp_eval.float().cpu()
            for b in range(B):
                sub_dir = save_root / f'{sample_idx:05d}'
                sub_dir.mkdir(parents=True, exist_ok=True)
                _to_pil_uint8_cpu(gt_cpu[b]).save(sub_dir / 'gt.png')
                _to_pil_uint8_cpu(inp_cpu[b]).save(sub_dir / 'input1.png')
                _to_pil_uint8_cpu(rec_cpu[b]).save(sub_dir / 'OURS1.png')
                ssim_b = float(ssim_vals[b])
                mse_b = float(mse_per[b].item())
                psnr_b = float(psnr_per[b].item())
                lpips_b = float(lpips_per[b].item())
                per_sample_rows.append([sample_idx, f'{ssim_b:.6f}', f'{mse_b:.8f}', f'{psnr_b:.6f}', f'{lpips_b:.6f}'])
                ssim_sum += ssim_b
                mse_sum += mse_b
                psnr_sum += psnr_b
                lpips_sum += lpips_b
                n_samples += 1
                sample_idx += 1
        del rec, rec_eval, gt_eval, inp_eval
        del gt, y, inp
        del mse_per, psnr_per, lpips_per
        del rec_cpu, gt_cpu, inp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ssim_avg = ssim_sum / max(n_samples, 1)
    mse_avg = mse_sum / max(n_samples, 1)
    psnr_avg = psnr_sum / max(n_samples, 1)
    lpips_avg = lpips_sum / max(n_samples, 1)
    metrics_txt = f'Samples: {n_samples}\nSSIM:   {ssim_avg:.6f}\nMSE:    {mse_avg:.8f}\nPSNR:   {psnr_avg:.6f} dB\nLPIPS:  {lpips_avg:.6f}\n'
    atomic_write_text(save_root / 'metrics.txt', metrics_txt)
    atomic_write_csv(save_root / 'per_sample_metrics.csv', per_sample_rows, header=['sample_idx', 'ssim', 'mse', 'psnr_db', 'lpips'])
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 512
    bayer = True
    model = Unrolled_net(num_groups=4, inner_steps=0, bayer=bayer).to(device)
    ckpt_path = '/media/lzy/research/hlj/Unfold_Ring/result of weight/cumm_ep150.pth'
    test_folder = '/media/lzy/research/hlj/Unfold_Ring/gray_512'
    BASE_OUT = script_dir() / 'outputs'
    EVAL_ROOT = BASE_OUT / 'result_cumm' / 'test_only'
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=True)
        if 'dim' in ckpt:
            pass
        if 'bayer' in ckpt:
            pass
    else:
        model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    se_r = torch.tensor([0.3318, 0.0081, 0.2914, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    se_g = torch.tensor([0.3318, 0.0081, 0.2914, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    se_b = torch.tensor([0.3318, 0.0081, 0.2914, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    psfW_dev = build_psfW_from_seidel(se_r, se_g, se_b, dim, bayer, device)
    psf_map_dev = build_psf_map_from_seidel(se_r, se_g, se_b, dim, device, k=34, kc=5, angle_bins=180, energy_frac=0.8)
    check_tensor(psfW_dev.detach().cpu(), 'psfW_dev')
    check_tensor(psf_map_dev.detach().cpu(), 'psf_map_dev')
    if bayer:
        test_ds = DeblurDataset(test_folder, psfW_dev)
    else:
        test_ds = GrayDeblurDataset(test_folder, psfW_dev)
    te_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    evaluate_and_save(model=model, psfW=psfW_dev, psf_map=psf_map_dev, loader=te_loader, device=device, save_root=str(EVAL_ROOT), bayer=bayer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
from pathlib import Path
from typing import List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from functions.PSF_compute import get_rdm_psfs, build_psf_feature_map
from model.Unroll import Unrolled_net
import numpy as np
from scipy.ndimage import median_filter

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

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']

def pil_to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    if img.mode not in ['RGB', 'L']:
        if 'A' in img.getbands():
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...] / 255.0
    else:
        arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).float()

def tensor_to_pil_uint8(t3: torch.Tensor) -> Image.Image:
    t = t3.detach().clamp(0, 1).cpu().numpy()
    if t.shape[0] == 1:
        arr = (t[0] * 255.0).round().astype('uint8')
        return Image.fromarray(arr, mode='L')
    elif t.shape[0] == 3:
        arr = (t.transpose(1, 2, 0) * 255.0).round().astype('uint8')
        return Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f'Unsupported channel count: {t.shape[0]}')

def center_crop_to_multiple(img_t: torch.Tensor, multiple: int=8) -> torch.Tensor:
    C, H, W = img_t.shape
    H2 = H // multiple * multiple
    W2 = W // multiple * multiple
    if H2 <= 0 or W2 <= 0:
        raise ValueError(f'Image too small: {(H, W)}')
    y0 = (H - H2) // 2
    x0 = (W - W2) // 2
    return img_t[:, y0:y0 + H2, x0:x0 + W2]

class RealBlurFolderDataset(Dataset):

    def __init__(self, root_or_file: str, bayer: bool=False, force_rgb: bool=True, crop_to_multiple: int=8, resize_to: int | None=None):
        self.path = Path(root_or_file)
        self.bayer = bayer
        self.force_rgb = force_rgb
        self.crop_to_multiple = crop_to_multiple
        self.resize_to = resize_to
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            files = sorted([p for p in self.path.rglob('*') if p.is_file() and is_image_file(p)])
        else:
            raise FileNotFoundError(f'Input path not found: {self.path}')
        if len(files) == 0:
            raise RuntimeError(f'No image files found in: {self.path}')
        self.files: List[Path] = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p)
        if self.force_rgb:
            img = img.convert('RGB')
        else:
            pass
        orig_hw = (img.height, img.width)
        if self.resize_to is not None:
            img = img.resize((self.resize_to, self.resize_to), Image.BICUBIC)
        x = pil_to_tensor_rgb01(img)
        if not self.bayer and x.shape[0] == 3:
            x = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
        if self.crop_to_multiple is not None and self.crop_to_multiple > 1:
            x = center_crop_to_multiple(x, self.crop_to_multiple)
        proc_hw = (x.shape[1], x.shape[2])
        meta = {'name': p.stem, 'path': str(p), 'orig_hw': orig_hw, 'proc_hw': proc_hw}
        return (x, meta)

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

def infer_and_save_realworld(model, psfW, psf_map, loader, device, save_root, bayer=True):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    psfW = psfW.detach()
    psf_map = psf_map.detach()
    log_lines = []
    log_lines.append('Real-world inference log')
    log_lines.append('')
    sample_idx = 0
    for x, meta in loader:
        if isinstance(meta, dict):
            name = meta['name'][0] if isinstance(meta['name'], list) else meta['name']
            src_path = meta['path'][0] if isinstance(meta['path'], list) else meta['path']
            orig_hw = meta['orig_hw'][0] if isinstance(meta['orig_hw'], list) else meta['orig_hw']
            proc_hw = meta['proc_hw'][0] if isinstance(meta['proc_hw'], list) else meta['proc_hw']
        else:
            name = f'{sample_idx:05d}'
            src_path = 'unknown'
            orig_hw = (-1, -1)
            proc_hw = (-1, -1)
        x = x.to(device, non_blocking=True)
        with torch.enable_grad():
            inp = x.detach().requires_grad_(True)
            rec_output = model(inp, psfW, psf_map)
            rec = rec_output[-1]
            del rec_output
        with torch.no_grad():
            rec_eval = rec.detach()
            inp_eval = inp.detach()
            B = rec_eval.shape[0]
            assert B == 1, 'real-world script默认 batch_size=1'
            out_dir = save_root / f'{sample_idx:05d}_{name}'
            out_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_pil_uint8(inp_eval[0]).save(out_dir / 'input_blur.png')
            tensor_to_pil_uint8(rec_eval[0]).save(out_dir / 'deblur_ours.png')
            torch.save({'input': inp_eval[0].cpu(), 'output': rec_eval[0].cpu(), 'meta': {'name': name, 'src_path': src_path, 'orig_hw': orig_hw, 'proc_hw': proc_hw}}, out_dir / 'result_tensor.pt')
            log_lines.append(f'[{sample_idx:05d}] {name}')
            log_lines.append(f'  src_path: {src_path}')
            log_lines.append(f'  orig_hw:  {orig_hw}')
            log_lines.append(f'  proc_hw:  {proc_hw}')
            log_lines.append(f'  save_dir: {str(out_dir)}')
            log_lines.append('')
        del rec, rec_eval, inp_eval, inp, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sample_idx += 1
    atomic_write_text(save_root / 'inference_log.txt', '\n'.join(log_lines))
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bayer = True
    dim = 512
    ckpt_path = '/media/lzy/research/hlj/Unfold_Ring/tests/outputs_gray_1/result_of_weight/cumm_ep040.pth'
    real_blur_path = '/media/lzy/research/hlj/Unfold_Ring/tests/input_real'
    BASE_OUT = script_dir() / 'outputs—new'
    OUT_ROOT = BASE_OUT / 'real_world_result'
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    se_r = torch.tensor([0.9159, 0.3318, 0.0081, 0.2914, 0.0, 0.0], device=device)
    se_g = torch.tensor([0.9159, 0.3318, 0.0081, 0.2914, 0.0, 0.0], device=device)
    se_b = torch.tensor([0.9159, 0.3318, 0.0081, 0.2914, 0.0, 0.0], device=device)
    psf_k = 34
    psf_kc = 5
    angle_bins = 180
    energy_frac = 0.8
    resize_to = dim
    model = Unrolled_net(num_groups=4, inner_steps=30, bayer=bayer).to(device)
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
    psfW_dev = build_psfW_from_seidel(se_r, se_g, se_b, dim, bayer, device)
    psf_map_dev = build_psf_map_from_seidel(se_r, se_g, se_b, dim, device, k=psf_k, kc=psf_kc, angle_bins=angle_bins, energy_frac=energy_frac)
    check_tensor(psfW_dev.detach().cpu(), 'psfW_dev')
    check_tensor(psf_map_dev.detach().cpu(), 'psf_map_dev')
    real_ds = RealBlurFolderDataset(root_or_file=real_blur_path, bayer=bayer, force_rgb=True, crop_to_multiple=8, resize_to=resize_to)
    real_loader = DataLoader(real_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=None)
    infer_and_save_realworld(model=model, psfW=psfW_dev, psf_map=psf_map_dev, loader=real_loader, device=device, save_root=str(OUT_ROOT), bayer=bayer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

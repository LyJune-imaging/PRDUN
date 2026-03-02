import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from functions.polar_trans import A_color

class DeblurDataset(Dataset):

    def __init__(self, gt_folder, psfW, poisson_range=(7000.0, 10000.0)):
        self.psfW = psfW.detach().cpu()
        self.paths = sorted([os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))])
        self.poisson_range = poisson_range

    def update_psf(self, new_psfW: torch.Tensor):
        self.psfW = new_psfW.detach().cpu()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        arr = np.asarray(img, np.float32) / 255.0
        gt = torch.from_numpy(arr).permute(2, 0, 1)
        with torch.no_grad():
            y = A_color(gt.unsqueeze(0), self.psfW, bayer=False).squeeze(0)
        return (gt, y.cpu())

class GrayDeblurDataset(Dataset):

    def __init__(self, img_folder: str, psfW: torch.Tensor):
        self.psfW = psfW
        self.device = psfW.device
        self.paths = sorted((os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.tif'))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        arr = np.asarray(img, np.float32) / 255.0
        gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        gt = torch.from_numpy(gray).unsqueeze(0)
        with torch.no_grad():
            inp = gt.unsqueeze(0).to(self.device)
            y = A_color(inp, self.psfW, bayer=False).squeeze(0)
            y = y.clamp(0.0, 1.0)
            peak = torch.empty(1, device=y.device).uniform_(10000.0, 13000.0).item()
            y_poisson = torch.poisson(y * peak) / peak
            y_noisy = y_poisson
            y = y_noisy.clamp(0.0, 1.0)
        return (gt, y.cpu())

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from functions.polar_trans import A_color as A
from pytorch_msssim import ssim as ssim_loss
from functions.PSF_compute import seidel2psfw
from Seldel_MLP.MLP import SeidelMLP

def to_numpy(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu()
    if arr.ndim == 2:
        return arr.numpy()
    elif arr.ndim == 3:
        return arr.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f'Unsupported tensor shape: {arr.shape}')
dim = 256
device = 'cuda'
img = plt.imread('') / 255.0
y_true = torch.tensor(img, dtype=torch.float32, device=device)
se_true = torch.tensor([0.8, 0.5, 0.6, 0.4, 0.3, 0.2], device=device)
psf_true = seidel2psfw(se_true, dim, diff=False)
y_blur = A(y_true, psf_true)
z0 = torch.randn(1, 100, device=device)
dip_net = SeidelMLP(noise_dim=100, hidden_dim=128, out_dim=6).to(device)
optimizer = optim.Adam(dip_net.parameters(), lr=0.0003)
criterion = nn.MSELoss()
torch.autograd.set_detect_anomaly(True)
criterion_mse = nn.MSELoss()
output_dir = 'epoch_snapshots'
num_epochs = 200
for epoch in range(0, num_epochs):
    dip_net.train()
    optimizer.zero_grad()
    se_pred = dip_net(z0).squeeze(0)
    psf_pred = seidel2psfw(se_pred, dim, diff=True)
    y_pred = A(y_true, psf_pred)
    l_mse = criterion_mse(y_pred, y_blur)
    l_l1 = F.l1_loss(y_pred, y_blur)
    yp = y_pred.unsqueeze(0) if y_pred.ndim == 3 else y_pred.unsqueeze(0).unsqueeze(0)
    yb = y_blur.unsqueeze(0) if y_blur.ndim == 3 else y_blur.unsqueeze(0).unsqueeze(0)
    l_ssim = 1 - ssim_loss(yp, yb, data_range=1.0, size_average=True)
    loss = l_mse + 0.5 * l_l1 + 0.5 * l_ssim
    loss.backward()
    optimizer.step()

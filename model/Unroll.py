import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from functions.polar_trans import AT_color, A_color
from model.Net import UNet_PMWA_PGDA

def replace_border_with_inner(x, rings=1):
    inner = x[:, :, rings:-rings, rings:-rings]
    filled = F.pad(inner, (rings, rings, rings, rings), mode='replicate')
    return filled

class Unrolled_net(nn.Module):

    def __init__(self, num_groups=2, inner_steps=3, bayer=False, res_scale=0.5):
        super().__init__()
        self.num_groups = num_groups
        self.inner = inner_steps
        self.bayer = bayer
        self.res_scale = res_scale
        self.spatial_nets = nn.ModuleList([UNet_PMWA_PGDA(input_resolution_hint=(256, 256), window_size=8, use_pmwa_out=True, heads_out=4, m_out=8, dec_resblocks=2, res_scale=0.5, pad_multiple=64) for _ in range(num_groups)])
        init_invL = 1.0
        init_rho = 0.0
        self.invL = nn.ParameterList([nn.Parameter(torch.tensor(init_invL), requires_grad=False) for _ in range(num_groups)])
        self.rho = nn.ParameterList([nn.Parameter(torch.tensor(init_rho), requires_grad=False) for _ in range(num_groups)])

    def forward(self, y, psfW, psf_feature):
        save_image(y[0], 'input.png')
        psfW = psfW.detach()
        x = y.clone()
        z = x.clone()
        rec_output = []
        if psf_feature.dim() == 3:
            psf_feature = psf_feature.unsqueeze(0).expand(z.size(0), -1, -1, -1).contiguous()
        elif psf_feature.dim() != 4:
            raise ValueError(f'psf_feature must be (30,H,W) or (B,30,H,W), got {psf_feature.shape}')
        for g in range(self.num_groups):
            invL_g = self.invL[g]
            rho_g = self.rho[g]
            for _ in range(self.inner):
                Az = A_color(z, psfW, bayer=self.bayer)
                grad = AT_color(Az - y, psfW, bayer=self.bayer)
                v = z - invL_g * grad
                z = v + rho_g * (v - x)
                x = v
                z = replace_border_with_inner(z, 1)
            if g == 0:
                save_image(z[0], 'ring.png')
            z = self.spatial_nets[g](z, psf_feature)
            if g == 3:
                save_image(z[0], 'net.png')
            rec_output.append(z)
        return rec_output

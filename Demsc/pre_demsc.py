import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibrateKernels(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

        def init_kernel(arr):
            return nn.Parameter(torch.tensor(arr, dtype=torch.float32, device=device).view(1, 1, 5, 5) / 8.0)
        G = [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]
        RG_red = [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]]
        RG_blue = [[0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0.5, 0, 5, 0, 0.5], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0]]
        RB = [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]
        BG_red, BG_blue, BR = (RG_blue, RG_red, RB)
        self.G_at_R = init_kernel(G)
        self.G_at_B = init_kernel(G)
        self.R_at_G_red = init_kernel(RG_red)
        self.R_at_G_blue = init_kernel(RG_blue)
        self.R_at_B = init_kernel(RB)
        self.B_at_G_red = init_kernel(BG_red)
        self.B_at_G_blue = init_kernel(BG_blue)
        self.B_at_R = init_kernel(BR)

    def forward(self, bayer):
        B, C, H, W = bayer.shape
        mono = bayer.sum(dim=1, keepdim=True)
        m = F.pad(mono, (2, 2, 2, 2), mode='reflect')
        G_R = F.conv2d(m, self.G_at_R)
        G_B = F.conv2d(m, self.G_at_B)
        R_Gr = F.conv2d(m, self.R_at_G_red)
        R_Gb = F.conv2d(m, self.R_at_G_blue)
        R_B = F.conv2d(m, self.R_at_B)
        B_Gr = F.conv2d(m, self.B_at_G_red)
        B_Gb = F.conv2d(m, self.B_at_G_blue)
        B_R = F.conv2d(m, self.B_at_R)
        ys = torch.arange(H, device=self.device).view(H, 1).expand(H, W)
        xs = torch.arange(W, device=self.device).view(1, W).expand(H, W)
        mask_R = ((ys % 2 == 0) & (xs % 2 == 0))[None, None].float().expand(B, 1, H, W)
        mask_Gr = ((ys % 2 == 0) & (xs % 2 == 1))[None, None].float().expand(B, 1, H, W)
        mask_Gb = ((ys % 2 == 1) & (xs % 2 == 0))[None, None].float().expand(B, 1, H, W)
        mask_B = ((ys % 2 == 1) & (xs % 2 == 1))[None, None].float().expand(B, 1, H, W)
        out = torch.zeros_like(bayer)
        out[:, 0:1] = bayer[:, 0:1] * mask_R + R_Gr * mask_Gr + R_Gb * mask_Gb + R_B * mask_B
        out[:, 1:2] = bayer[:, 1:2] * (mask_Gr + mask_Gb) + G_R * mask_R + G_B * mask_B
        out[:, 2:3] = bayer[:, 2:3] * mask_B + B_Gr * mask_Gr + B_Gb * mask_Gb + B_R * mask_R
        return out

class RefineNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
calib = CalibrateKernels(device).to(device)
refine = RefineNet().to(device)
checkpoint = torch.load('/media/lzy/research/hlj/Unfold_Ring/dl_models/Demsc/mhc_calib_refine.pth', map_location=device, weights_only=True)
calib.load_state_dict(checkpoint['calib'])
refine.load_state_dict(checkpoint['refine'])
calib.eval()
refine.eval()

def demosaic_mhc(bayer):
    with torch.no_grad():
        out_calib = calib(bayer)
        out_ref = refine(out_calib)
    return out_ref

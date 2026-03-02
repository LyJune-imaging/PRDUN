import torch

def bayer_mosaic(img, pattern='RGGB'):
    if img.ndim == 3:
        out = torch.zeros_like(img)
        if pattern == 'RGGB':
            out[0, 0::2, 0::2] = img[0, 0::2, 0::2]
            out[1, 0::2, 1::2] = img[1, 0::2, 1::2]
            out[1, 1::2, 0::2] = img[1, 1::2, 0::2]
            out[2, 1::2, 1::2] = img[2, 1::2, 1::2]
        else:
            raise NotImplementedError('Pattern only supports RGGB')
        return out
    elif img.ndim == 4:
        out = torch.zeros_like(img)
        if pattern == 'RGGB':
            out[:, 0, 0::2, 0::2] = img[:, 0, 0::2, 0::2]
            out[:, 1, 0::2, 1::2] = img[:, 1, 0::2, 1::2]
            out[:, 1, 1::2, 0::2] = img[:, 1, 1::2, 0::2]
            out[:, 2, 1::2, 1::2] = img[:, 2, 1::2, 1::2]
        else:
            raise NotImplementedError('Pattern only supports RGGB')
        return out
    else:
        raise ValueError('Only supports [3,H,W] or [B,3,H,W]')

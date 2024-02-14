import torch
from typing import Union, List, Tuple
from diffusers import DiffusionPipeline
import numpy as np

def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def _get_pattern(shape, pipe, w_pattern='ring', w_seed=999999):
    g = torch.Generator(device=pipe.device)
    g.manual_seed(w_seed)
    gt_init = pipe.get_random_latents(generator=g)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = gt_patch.clone().detach()
        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch

# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str) -> torch.Tensor:
def get_noise(shape: Union[torch.Size, List, Tuple], pipe) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
    # for now we hard code all hyperparameters
    w_seed = 999999 # seed for key
    w_channel = 0 # id for watermarked channel
    w_radius = 10 # watermark radius
    w_pattern = 'rand' # watermark pattern

    # get watermark key and mask
    np_mask = _circle_mask(shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask).to(pipe.device)
    w_mask = torch.zeros(shape, dtype=torch.bool).to(pipe.device)
    w_mask[:, w_channel] = torch_mask
    
    w_key = _get_pattern(shape, pipe, w_pattern=w_pattern, w_seed=w_seed).to(pipe.device)

    # inject watermark
    init_latents = pipe.get_random_latents()
    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_mask

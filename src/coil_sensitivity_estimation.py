from itertools import product
import numpy as np
import torch
import scipy
from torch.fft import ifftshift, ifft, fftshift, fft
import torch.nn.functional as F
from tqdm import tqdm
import einops as eo
import torchkbnufft as tkbn

from src import computation as comp

def lowk_xy(kspace_data, adjnufft_ob, kspace_traj, batch_size=2, device = torch.device('cpu'), **kwargs):
    spoke_len = kspace_data.shape[-1]
    W = comp.hamming_filter(nonzero_width_percent=0.03, width = spoke_len)
    spoke_lowpass_filter_xy = torch.from_numpy(W)

    @comp.batch_process(batch_size=batch_size, device=device, batch_dim = 0)
    def apply_filter_and_nufft(kspace_data, filter, ktraj ):
        kspace_data = filter*kspace_data
        kspace_data = comp.ifft_1D(kspace_data,dim=1)
        # TODO why we need flip?
        kspace_data = torch.flip(kspace_data, dims=(1,))
        kspace_data = kspace_data/kspace_data.abs().max()
        kspace_data = eo.rearrange(kspace_data, 'ch_num slice_num spoke_num spoke_len -> slice_num ch_num (spoke_num spoke_len)').contiguous()
        img_dc = adjnufft_ob.forward(kspace_data, ktraj)
        img_dc = eo.rearrange(img_dc, 'slice_num ch_num h w -> ch_num slice_num h w')
        # print(img_dc.shape)
        return img_dc

    coil_sens = apply_filter_and_nufft(
        kspace_data, 
        filter = spoke_lowpass_filter_xy,
        ktraj=eo.rearrange(kspace_traj,'c spoke_num spoke_len -> c (spoke_num spoke_len)'),)
    img_sens_SOS = torch.sqrt(eo.reduce(coil_sens.abs()**2, 'ch_num slice_num height width -> () slice_num height width', 'sum'))
    coil_sens = coil_sens/img_sens_SOS
    return coil_sens


def lowk_xyz(kspace_data, adjnufft_ob, kspace_traj,  batch_size=2, device = torch.device('cpu'), **kwargs):
    # "need to be used before kspace z axis ifft"
    spoke_len = kspace_data.shape[-1]
    slice_num = kspace_data.shape[1]
    W = comp.hamming_filter(nonzero_width_percent=0.03, width = spoke_len)
    spoke_lowpass_filter_xy = torch.from_numpy(W)
    Wz = comp.hamming_filter(nonzero_width_percent=0.03, width = slice_num)
    spoke_lowpass_filter_z = torch.from_numpy(Wz)

    @comp.batch_process(batch_size=batch_size, device=device)
    def apply_filter_and_nufft(kspace_data, filter_xy,filter_z,  ktraj):
        kspace_data = filter_xy*kspace_data
        kspace_data = eo.einsum(filter_z, kspace_data,'b, a b c d -> a b c d')
        kspace_data = comp.ifft_1D(kspace_data,dim=1)
        # TODO why we need flip?
        kspace_data = torch.flip(kspace_data, dims=(1,))
        kspace_data = kspace_data/kspace_data.abs().max()
        kspace_data = eo.rearrange(kspace_data, 'ch_num slice_num spoke_num spoke_len -> slice_num ch_num (spoke_num spoke_len)').contiguous()
        img_dc = adjnufft_ob.forward(kspace_data, ktraj)
        img_dc = eo.rearrange(img_dc, 'slice_num ch_num h w -> ch_num slice_num h w')
        return img_dc

    coil_sens = apply_filter_and_nufft(
        kspace_data, 
        filter_xy = spoke_lowpass_filter_xy, filter_z = spoke_lowpass_filter_z,
        ktraj=eo.rearrange(kspace_traj,'c spoke_num spoke_len -> c (spoke_num spoke_len)'),)
    img_sens_SOS = torch.sqrt(eo.reduce(coil_sens.abs()**2, 'ch_num slice_num height width -> () slice_num height width', 'sum'))
    coil_sens = coil_sens/img_sens_SOS
    return coil_sens


def lowk_xyz_per_phase(kspace_data, adjnufft_ob, kspace_traj,  batch_size=2, device = torch.device('cpu'), **kwargs):
    args = kwargs['args']
    t = kwargs['t']
    ph = kwargs['ph']
    # kspace_data_ = 
    # coil_sens = torch.zeros((args.contra_num, args.phase_num, args.ch_num, args.last_slice-args.first_slice,args.im_size[0],args.im_size[1]),dtype=torch.complex128)
    # for t,ph in product(range(args.contra_num),range(args.phase_num)):
    coil_sens = lowk_xyz(kspace_data[t,ph], adjnufft_ob, args.ktraj[t,ph], batch_size = batch_size*170)
    return coil_sens
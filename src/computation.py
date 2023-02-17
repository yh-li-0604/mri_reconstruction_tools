from asyncio import constants
from itertools import product
import numpy as np
import torch
import scipy
from torch.fft import ifftshift, ifft, fftshift, fft
import torch.nn.functional as F
from tqdm import tqdm
import einops as eo
import torchkbnufft as tkbn

from src.io_utils import *
from src.twix_metadata_def import *

def batch_process(batch_size:int, device:torch.device, batch_dim = 0):
    def Inner(func):
        def process(*args, **kwargs):
            outputs = []
            kwargs_input = dict((k,v.to(device)) if isinstance(v,torch.Tensor) else (k,v) for k,v in kwargs.items())
            args_batched = [ torch.split(data,batch_size,batch_dim) for data in args ]
            batch_num = len(args_batched[0])
            for batch_idx in tqdm(range(batch_num)):
                args_input = (data[batch_idx].to(device) for data in args_batched)
                outputs.append(func(*args_input, **kwargs_input).cpu())
            outputs = torch.cat(outputs,dim=batch_dim)
            for k,v in kwargs_input.items():
                if isinstance(v,torch.Tensor):
                    v.cpu()
            return outputs
        return process 
    return Inner

def hamming_filter(nonzero_width_percent:float, width:int)->np.ndarray:
    nonzero_width = round(width*nonzero_width_percent) 
    pad_width_L = round((width-nonzero_width)//2 )
    pad_width_R = width-nonzero_width-pad_width_L
    hamming_weights = np.float32(np.hamming(nonzero_width))
    W = np.pad(hamming_weights,pad_width=(pad_width_L,pad_width_R))
    return W

def tuned_and_robust_estimation(navigator: np.ndarray, percentW: float, Fs, FOV, ndata, device = torch.device('cuda')):
    '''
    return channel and rotation index and generated curve
    '''
    col_num, line_num, ch_num = navigator.shape

    # To reduce noise, the navigator k-space data were apodized using a Hamming window.
    W = hamming_filter(percentW/100, col_num)
    W = eo.repeat(W, 'col_num -> col_num line_num ch_num',
                  line_num=line_num, ch_num=ch_num)

    # New quality metric block begin
    N = navigator.shape[1]
    f = torch.linspace(-0.5*Fs, 0.5*Fs-Fs/N, steps=N,device=device)
    # compute the ifft of weighted navigator, using the representation in CAPTURE paper
    # col_num->x, line_num->n, ch_num->i, tuning_num->m
    K_weighted = torch.from_numpy(W*navigator).to(f.device)
    projections = fftshift(
        ifft(ifftshift(K_weighted, dim=0), dim=0), dim=0)  # shape is x n i

    # shape is m=100
    phase_rotation_factors = torch.exp(-1j*2*torch.pi *
                                       torch.arange(1, 101, device=f.device)/100)
    r = torch.empty(
        (projections.shape[1], projections.shape[2], 100), device=f.device)
    for m in range(100):
        r[:, :, m] = torch.argmax(
            (phase_rotation_factors[m]*projections[:, :, :]).real, dim=0)
    # A = torch.einsum('xni,m->xnim',projections,phase_rotation_factors).real # np.multiply.outer(projections, phase_rorate..)
    # r = torch.argmax(A,dim=0).to(torch.double)+1 # 'x n i m -> n i m'
    R = torch.abs(
        fftshift(fft(r-eo.reduce(r, 'n i m -> i m', 'mean'), dim=0), dim=0))

    lowfreq_integral = eo.reduce(
        R[(torch.abs(f) < 0.5) * (torch.abs(f) > 0.1)], 'f i m -> i m', 'sum')
    highfreq_integral = eo.reduce(R[torch.abs(f) > 0.8], 'f i m -> i m', 'sum')
    r_range = eo.reduce(r, 'n i m -> i m', 'max') - \
        eo.reduce(r, 'n i m -> i m', 'min')
    lower_bound = torch.full_like(r_range, 30/(FOV/(ndata/2)))
    # what does this FOV/ndata use for
    determinator = torch.maximum(r_range, lower_bound)
    Q = lowfreq_integral/highfreq_integral/determinator
    Q_np = Q.numpy(force=True)  # faster than matlab version 10x

    i_max, m_max = np.unravel_index(np.argmax(Q_np), Q_np.shape)
    # projection_max = projections[:, :, i_max]
    r_max = r[:, i_max, m_max].numpy(force=True)
    
    # new quality metric block end

    #filter high frequency signal
    b = scipy.signal.firwin(
        12, 1/(Fs/2), window="hamming", pass_zero='lowpass')
    a = 1
    r_max_low_pass = scipy.signal.filtfilt(b, a, r_max)
    r_max_SG = scipy.signal.filtfilt(
        b, a, scipy.signal.savgol_filter(r_max, 5, 1))
    r_max_filtered = r_max_low_pass.copy()
    r_max_filtered[0:10], r_max_filtered[-10:] = r_max_SG[0:10], r_max_SG[-10:]

    return i_max, m_max, torch.from_numpy(r_max_filtered)

def centralize_kspace(kspace_data, acquire_length, center_idx_in_acquire_lenth, full_length, dim)->torch.Tensor:
    # center_in_acquire_length is index, here +1 to turn into quantity
    # print(kspace_data[0,:,0,320])
    front_padding = round(full_length / 2 - (center_idx_in_acquire_lenth+1))
    # the dc point can be located at length/2 or length/2+1, when length is even, cihat use length/2+1
    front_padding += 1
    pad_length = [ 0 for i in range(2*len(kspace_data.shape))]
    pad_length[dim*2+1], pad_length[dim*2] = front_padding, full_length-acquire_length-front_padding
    pad_length.reverse()
    # torch.nn.functional.pad() are using pad_lenth in a inverse way. (pad_front for axis -1,pad_back for axis -1, pad_front for axis -2, pad_back for axis-2 ......)
    kspace_data_ = F.pad(kspace_data, pad_length, mode='constant') # default constant is 0
    return kspace_data_

def ifft_1D(kspace_data,dim = -1):
    return fftshift(ifft(ifftshift(kspace_data, dim=dim),dim=dim),dim=dim )

def generate_golden_angle_radial_spokes_kspace_trajctory(spokes_num, spoke_length):
    # create a k-space trajectory
    KWIC_GOLDENANGLE = 180*(np.sqrt(5)-1)/2#111.246117975
    k = torch.linspace(-0.5, 0.5-1/spoke_length,spoke_length)
    k[spoke_length//2] = 0
    A = torch.arange(spokes_num)*torch.pi*KWIC_GOLDENANGLE/180
    kx = torch.outer(torch.cos(A),k)
    ky = torch.outer(torch.sin(A),k)
    ktraj = torch.stack((kx, ky), dim=0)
    return ktraj



def data_binning(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase):
    spoke_len = data.shape[-1]
    output = eo.rearrange(
        data, 
        '... (t spokes_per_contra) spoke_len -> ... t spokes_per_contra spoke_len ',
        t = contrast_num,
        spokes_per_contra = spokes_per_contra
        )
    output = output.gather(
        dim=-2, 
        index = 
        eo.repeat(
            sorted_r_idx, 
            't spokes_per_contra -> t spokes_per_contra spoke_len',
            # t = nContrasts,
            spokes_per_contra = spokes_per_contra,
            spoke_len = spoke_len).expand_as(output)
    )
    output = eo.rearrange(
        output, 
        '... t (ph spoke) spoke_len -> t ph ...  spoke spoke_len',
        ph = phase_num,
        spoke = spokes_per_phase)
    return output

def recon_adjnufft(kspace_data, kspace_traj, adjnufft_ob, density_compensation_func):
    kspace_density_compensation = density_compensation_func(
            kspace_traj=kspace_traj,
            im_size=adjnufft_ob.im_size.numpy(force=True),
            grid_size=adjnufft_ob.grid_size.numpy(force=True))
    kspace_data = eo.rearrange(
        kspace_data*kspace_density_compensation,
        '... ch slice spoke spoke_len-> ... slice ch (spoke spoke_len)')
    kspace_traj = eo.rearrange(
        kspace_traj,
        '... c spoke spoke_len -> ... c (spoke spoke_len)') # c stands for complex channel
    img = adjnufft_ob.forward((kspace_data).contiguous(), kspace_traj,norm='ortho')

    img = eo.rearrange(img,'slice ch h w-> ch slice h w')
    return img

def polygon_area(vertices):
    '''
    vertice are tensor, vertices_num x dimensions(2)
    '''
    x,y = vertices[:,0],vertices[:,1]
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = torch.dot(x[:-1], y[1:]) - torch.dot(y[:-1], x[1:])
    return 0.5*torch.abs(main_area + correction)

def normalization(img):
    return (img-img.mean())/img.std()
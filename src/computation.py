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
    nonzero_width = int(width*nonzero_width_percent) 
    pad_width_L = int((width-nonzero_width)//2 )
    pad_width_R = int(width-nonzero_width-pad_width_L)
    hamming_weights = np.hamming(nonzero_width)
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
    # print(type( r_max ))
    
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

def centralize_kspace(kspace_data, acquire_length, center_in_acquire_lenth, full_length, dim)->torch.Tensor:
    diff_kdata = int(full_length // 2  - (center_in_acquire_lenth+1)) 
    #here center_in_acquire_length is index, here +1 to turn into quantity
    pad_length = [ 0 for i in range(2*len(kspace_data.shape))]
    # pad_length[dim*2], pad_length[dim*2+1] = diff_kdata, full_length-acquire_length-diff_kdata+1
    pad_length[dim*2], pad_length[dim*2+1] = diff_kdata, full_length-acquire_length-diff_kdata
    pad_length.reverse()
    # torch.nn.functional.pad() are using pad_lenth in a inverse way. (pad_front for axis -1,pad_back for axis -1, pad_front for axis -2, pad_back for axis-2 ......)
    kspace_data_ = F.pad(kspace_data, pad_length, mode='constant') # default constant is 0
    # kspace_data_ = torch.zeros(int(ndata), ntviews-max(nPhases,10), nslc_f, ch_num, dtype=torch.complex128)
    print(kspace_data_.shape)
    print(kspace_data.shape)
    print(full_length,acquire_length)
    return kspace_data_

def ifft_1D(kspace_data,dim = -1):
    return fftshift(ifft(ifftshift(kspace_data, dim=dim),dim=dim),dim=dim )

def generate_golden_angle_radial_spokes_kspace_trajctory(spokes_num, spoke_length):
    # create a k-space trajectory
    ga = torch.tensor(np.deg2rad(180 / ((1 + np.sqrt(5)) / 2)))
    kx = torch.zeros(spokes_num, spoke_length)
    ky = torch.zeros_like(kx)
    ky[0, :] = torch.linspace(-torch.pi, torch.pi, spoke_length)
    for i in range(1, spokes_num):
        kx[i, :] = torch.cos(ga) * kx[i - 1, :] - torch.sin(ga) * ky[i - 1, : ]
        ky[i, :] = torch.sin(ga) * kx[ i - 1,:] + torch.cos(ga) * ky[ i - 1, :]
    ktraj = torch.stack((ky, kx), dim=0)
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


def recon_adjnufft(kspace_data, smaps, kspace_traj, adjnufft_ob, density_compensation_func):
    # print(kspace_traj.shape,kspace_data.shape,smaps.shape)
    # print(kspace_data.shape)
    kspace_data = eo.rearrange(
        kspace_data,
        '... ch slice spoke spoke_len-> ... slice ch (spoke spoke_len)')
    kspace_traj = eo.rearrange(
        kspace_traj,
        '... c spoke spoke_len -> ... c (spoke spoke_len)') # c stands for complex channel
    smaps = eo.rearrange(
        smaps,
        '... ch slice h w-> ... slice ch h w')  
    # print(kspace_data.shape,kspace_traj.shape,smaps.shape)
    # k_space_density_compensation, it is flattened due to nufft, now recover it
    kspace_density_compensation = density_compensation_func(
            ktraj=kspace_traj,
            im_size=adjnufft_ob.im_size.numpy(force=True),
            grid_size=adjnufft_ob.grid_size.numpy(force=True))
    # print(kspace_density_compensation.shape)
    # img = adjnufft_ob.forward((kspace_data*kspace_density_compensation).contiguous(), kspace_traj, smaps=smaps)
    # print(kspace_data.shape,kspace_density_compensation.shape,kspace_traj.shape,smaps.shape)
    img = adjnufft_ob.forward((kspace_data*kspace_density_compensation).contiguous(), kspace_traj, smaps=smaps)
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


# def MCNUFFT(kspace_data, 
#     kspace_traj,
#     adjnufft_ob,coil_sensitivity_estimator
#     contrast_num,
#     phase_num,
#     slice_num,
#     first_slice,
#     last_slice,
#     im_size,
#     batch_size=80, 
#     device = torch.device('cpu')):
    '''
    kspace_data shape: t ph ch slice spoke spoke_len
    sensitivity_map shape: t ph ch slice h 
    kspace_traj shape: t ph c spoke spoke_len
    '''
    # kspace_data = eo.rearrange(
    #     kspace_data,
    #     '... ch slice spoke spoke_len-> ... slice ch (spoke spoke_len)')
    # kspace_traj = eo.rearrange(
    #     kspace_traj,
    #     '... c spoke spoke_len -> ... c (spoke spoke_len)') # c stands for complex channel
    # smaps, ktraj, kspace_data,
    # must change axis slice and ch outside, 
    # because we can only batch operation on slice axis, not ch axis, later will lead to uncomplete sensitivity compensate.

    # @batch_process(batch_size = 8, device = device)

    # img = torch.zeros((contrast_num,phase_num,last_slice-first_slice,im_size[0],im_size[1]),dtype=torch.complex128)

    # for t,ph in product(range(contrast_num),range(phase_num)):
    #     print('NUFFT for contrast:{}, phase:{}'.format(t, ph))
    #     sensitivity_map = csecoil_sensitivity_estimator[t,ph][:,args.first_slice:args.last_slice]
    #     output = recon_adjnufft(
    #             kspace_data[t,ph,args.first_slice:args.last_slice],
    #             sensitivity_map[args.first_slice:args.last_slice],
    #             kspace_traj = kspace_traj[t,ph],
    #             adjnufft_ob=adjnufft_ob
    #             )
    #     img[t,ph,:,:,:] = eo.reduce(output, 'slice ch w h -> slice w h', 'sum')
    # print('MCNUFFT reconstruction finished')
    # img = (img-img.mean())/img.std()
    # return img



    # if args.coil_sensitivity_estimation_func_outside:
    #     sensitivity_map = args.coil_sensitivity_estimation_func_outside(args_sens.kspace_data, adjnufft_ob, args_sens.ktraj,
    #         batch_size=2, device=device)
    #     sensitivity_map = eo.rearrange(
    #             sensitivity_map,
    #             '... ch slice h w-> ... slice ch h w')  
    # kspace_data = eo.rearrange(
    #     args.kspace_data,
    #     '... ch slice spoke spoke_len-> ... slice ch (spoke spoke_len)')
    # kspace_traj = eo.rearrange(
    #     args.ktraj,
    #     '... c spoke spoke_len -> ... c (spoke spoke_len)') # c stands for complex channel
    # # smaps, ktraj, kspace_data,
    # # must change axis slice and ch outside, 
    # # because we can only batch operation on slice axis, not ch axis, later will lead to uncomplete sensitivity compensate.

    # @batch_process(batch_size = batch_size, device=device)
    # def recon_adjnufft(kspace_data, smaps, ktraj, adjnufft_ob):
    #     img_dc = adjnufft_ob.forward(kspace_data.contiguous(), ktraj, smaps=smaps)
    #     # img_dc = eo.rearrange(img_dc,'slice ch h w-> ch slice h w')
    #     return img_dc

    # img_nufft = torch.zeros((args.contra_num,args.phase_num,args.last_slice-args.first_slice,args.im_size[0],args.im_size[1]),dtype=torch.complex128)

    # for t,ph in product(range(args.contra_num),range(args.phase_num)):
    #     print('NUFFT for contrast:{}, phase:{}'.format(t, ph))
    #     if args.coil_sensitivity_estimation_func_inside:
    #         sensitivity_map = args.coil_sensitivity_estimation_func_inside(
    #             args_sens.kspace_data[t,ph], adjnufft_ob, args_sens.ktraj[t,ph],
    #             batch_size=80, device=device)
    #         #TODO exemely slow, optimize this
    #         sensitivity_map = eo.rearrange(
    #             sensitivity_map,
    #             '... ch slice h w-> ... slice ch h w')  
    #     output = recon_adjnufft(
    #             kspace_data[t,ph,args.first_slice:args.last_slice],
    #             sensitivity_map[args.first_slice:args.last_slice],
    #             ktraj = kspace_traj[t,ph],
    #             adjnufft_ob=adjnufft_ob
    #             )#[:,:,int(img_size[0]/4):3*int(img_size[0]/4),int(img_size[1]/4):3*int(img_size[1]/4)]
    #     img_nufft[t,ph,:,:,:] = eo.reduce(output, 'slice ch w h -> slice w h', 'sum')
    '''
        for slice_idx in tqdm(range(slice_range[0],slice_range[1])):
            if len(sensitivity_map.shape) == 4:
                output = recon_adjnufft(
                    eo.rearrange(
                        kspace_density_compensation*kspace_data[slice_idx],
                        'ch t ph spoke spoke_len-> (t ph) ch (spoke spoke_len)'
                    ),
                    eo.rearrange(kspace_traj,
                    'c t ph spoke spoke_len -> (t ph) c (spoke spoke_len)'), # c stands for complex channel
                    # k_space_density_compensation = k_space_density_compensation, 
                    smaps=sensitivity_map[slice_idx],
                    adjnufft_ob=adjnufft_ob
                    )#[:,:,int(img_size[0]/4):3*int(img_size[0]/4),int(img_size[1]/4):3*int(img_size[1]/4)]
            elif len(sensitivity_map.shape) == 6:
                output = recon_adjnufft(
                    eo.rearrange(
                        kspace_density_compensation*kspace_data[slice_idx],
                        'ch t ph spoke spoke_len-> (t ph) ch (spoke spoke_len)'
                    ),
                    eo.rearrange(kspace_traj,
                    'c t ph spoke spoke_len -> (t ph) c (spoke spoke_len)'), # c stands for complex channel
                    eo.rearrange(sensitivity_map,
                    't ph slice_num ch_num h w-> (t ph) slice_num ch_num h w)')[:,slice_idx], # c stands for complex channel
                    adjnufft_ob=adjnufft_ob
                    )#[:,:,int(img_size[0]/4):3*int(img_size[0]/4),int(img_size[1]/4):3*int(img_size[1]/4)]
    '''
  
def normalization(img):
    return (img-img.mean())/img.std()
    # img_nufft = eo.rearrange(img_nufft, '(t ph) d w h -> t ph d w h', t = contrast_num, ph=phase_num)
    # print('MCNUFFT reconstruction finished')
    # return img_nufft
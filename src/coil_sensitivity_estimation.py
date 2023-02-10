from dataclasses import field
import numpy as np
import torch
import einops as eo
import torchkbnufft as tkbn

from src import computation as comp


def lowk_xy(kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio=0.05, batch_size=2, device=torch.device('cpu')):
    spoke_len = kspace_data.shape[-1]
    W = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio, width=spoke_len)
    spoke_lowpass_filter_xy = torch.from_numpy(W)

    @comp.batch_process(batch_size=batch_size, device=device, batch_dim=0)
    def apply_filter_and_nufft(kspace_data, filter, ktraj):
        kspace_data = filter*kspace_data
        kspace_data = comp.ifft_1D(kspace_data, dim=1)
        # TODO why we need flip?
        kspace_data = torch.flip(kspace_data, dims=(1,))
        kspace_data = kspace_data/kspace_data.abs().max()
        kspace_data = eo.rearrange(
            kspace_data, 'ch_num slice_num spoke_num spoke_len -> slice_num ch_num (spoke_num spoke_len)').contiguous()
        # interp_mats = tkbn.calc_tensor_spmatrix(ktraj,im_size=adjnufft_ob.im_size.numpy(force=True))
        img_dc = adjnufft_ob.forward(kspace_data, ktraj,norm='ortho')
        img_dc = eo.rearrange(
            img_dc, 'slice_num ch_num h w -> ch_num slice_num h w')
        # print(img_dc.shape)
        return img_dc

    coil_sens = apply_filter_and_nufft(
        kspace_data,
        filter=spoke_lowpass_filter_xy,
        ktraj=eo.rearrange(kspace_traj, 'c spoke_num spoke_len -> c (spoke_num spoke_len)'),)
    
    coil_sens = coil_sens[:,:,spoke_len//2-spoke_len//4:spoke_len//2+spoke_len//4,spoke_len//2-spoke_len//4:spoke_len//2+spoke_len//4]
    # coil_sens = torch.from_numpy(coil_sens)
    img_sens_SOS = torch.sqrt(
        eo.reduce(
            coil_sens.abs()**2, 
            'ch_num slice_num height width -> () slice_num height width', 'sum'))
    coil_sens = coil_sens/img_sens_SOS
    coil_sens[torch.isnan(coil_sens)] = 0 # optional
    coil_sens /= coil_sens.abs().max()
    return coil_sens 


def lowk_xyz(kspace_data, kspace_traj,  adjnufft_ob, hamming_filter_ratio=[0.05,0.2], batch_size=2, device=torch.device('cpu'), **kwargs):
    # "need to be used before kspace z axis ifft"
    spoke_len = kspace_data.shape[-1]
    slice_num = kspace_data.shape[1]
    W = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio[0], width=spoke_len)
    spoke_lowpass_filter_xy = torch.from_numpy(W)
    Wz = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio[1], width=slice_num)
    spoke_lowpass_filter_z = torch.from_numpy(Wz)

    @comp.batch_process(batch_size=batch_size, device=device)
    def apply_filter_and_nufft(kspace_data, filter_xy, filter_z,  ktraj):
        kspace_data = filter_xy*kspace_data
        kspace_data = eo.einsum(filter_z, kspace_data, 'b, a b c d -> a b c d')
        kspace_data = comp.ifft_1D(kspace_data, dim=1)
        # TODO why we need flip?
        kspace_data = torch.flip(kspace_data, dims=(1,))
        kspace_data = kspace_data/kspace_data.abs().max()
        kspace_data = eo.rearrange(
            kspace_data, 'ch_num slice_num spoke_num spoke_len -> slice_num ch_num (spoke_num spoke_len)').contiguous()
        img_dc = adjnufft_ob.forward(kspace_data, ktraj)
        img_dc = eo.rearrange(
            img_dc, 'slice_num ch_num h w -> ch_num slice_num h w')
        return img_dc

    coil_sens = apply_filter_and_nufft(
        kspace_data,
        filter_xy=spoke_lowpass_filter_xy, filter_z=spoke_lowpass_filter_z,
        ktraj=eo.rearrange(kspace_traj, 'c spoke_num spoke_len -> c (spoke_num spoke_len)'),)
    coil_sens = coil_sens[:,:,spoke_len//2-spoke_len//4:spoke_len//2+spoke_len//4,spoke_len//2-spoke_len//4:spoke_len//2+spoke_len//4]
    img_sens_SOS = torch.sqrt(eo.reduce(coil_sens.abs(
    )**2, 'ch_num slice_num height width -> () slice_num height width', 'sum'))
    coil_sens = coil_sens/img_sens_SOS
    coil_sens[torch.isnan(coil_sens)] = 0 # optional
    coil_sens /= coil_sens.abs().max()
    return coil_sens


# def lowk_xyz_per_phase(kspace_data,  kspace_traj,adjnufft_ob, current_contrast, current_phase, batch_size=2, device = torch.device('cpu'), **kwargs):
#     coil_sens = lowk_xyz(kspace_data[current_contrast,current_phase], adjnufft_ob, kspace_traj[current_contrast,current_phase], batch_size = batch_size*170)
#     return coil_sens


class CoilSensitivityEstimator:
    def __init__(self, kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio, batch_size, device) -> None:
        self.device = device
        self.op = adjnufft_ob
        self.kspace_data = kspace_data
        self.kspace_traj = kspace_traj
        self.hamming_filter_ratio = hamming_filter_ratio
        self.batch_size = batch_size
        self.coil_sens = field(default_factory=torch.Tensor)

    def __getitem__(self, key):
        return self.coil_sens[key]


class Lowk_2D_CSE(CoilSensitivityEstimator):
    def __init__(self, kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio=0.05, batch_size=2, device=torch.device('cpu')) -> None:
        super().__init__(kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio, batch_size, device)
        # stacked_spoke_kspace_traj = eo.rearrange(kspace_traj,'t ph c spoke_num spoke_len -> c (t ph spoke_num) spoke_len')
        self.coil_sens = lowk_xy(
            kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio, batch_size=batch_size, device=device)

    def __getitem__(self, key):
        current_contrast = key[0]
        current_phase = key[1]
        return super().__getitem__(key[2:])


class Lowk_3D_CSE(CoilSensitivityEstimator):
    def __init__(self, kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio=[0.05,0.5],  batch_size=2, device=torch.device('cpu')) -> None:
        super().__init__(kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio, batch_size, device)
        self.coil_sens = lowk_xyz(
            kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio,batch_size=batch_size, device=device)

    def __getitem__(self, key):
        return super().__getitem__(key[2:])


class Lowk_5D_CSE(CoilSensitivityEstimator):
    def __init__(self, kspace_data, kspace_traj, adjnufft_ob, args, density_compensation_func, hamming_filter_ratio=[0.05,0.5], batch_size=2, device=torch.device('cpu')) -> None:
        super().__init__(kspace_data, kspace_traj, adjnufft_ob, hamming_filter_ratio, batch_size, device)
        self.kspace_traj,  self.kspace_data = map(
            comp.data_binning,
            [kspace_traj,  kspace_data],
            [args.sorted_r_idx]*2, [args.contra_num]*2,
            [args.spokes_per_contra]*2, [args.phase_num]*2,
            [args.spokes_per_phase]*2)
        self.density_compensation_func = density_compensation_func

    def __getitem__(self, key):
        current_contrast = key[0]
        current_phase = key[1]
        kspace_traj = self.kspace_traj[current_contrast, current_phase]
        kspace_density_compensation = self.density_compensation_func(
            kspace_traj=kspace_traj,
            im_size=self.op.im_size.numpy(force=True),
            grid_size=self.op.grid_size.numpy(force=True))
        return lowk_xyz(self.kspace_data[
            current_contrast, current_phase],
            kspace_traj,
            self.op, self.hamming_filter_ratio, batch_size=self.batch_size, device=self.device)

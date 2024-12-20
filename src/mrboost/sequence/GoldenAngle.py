# %%
from dataclasses import dataclass, field
from typing import Callable, Dict

import einx
import torch

# from icecream import ic
from plum import dispatch

from mrboost import computation as comp
from mrboost.coil_sensitivity_estimation import (
    get_csm_lowk_xy,
)
from mrboost.density_compensation import (
    voronoi_density_compensation,
)

# from mrboost.io_utils import *
from mrboost.sequence.boilerplate import ReconArgs


@dataclass
class GoldenAngleArgs(ReconArgs):
    adjnufft: Callable = field(init=False)
    nufft: Callable = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.adjnufft = lambda x, y: comp.nufft_adj_2d(x, y, self.im_size)
        self.nufft = lambda x, y: comp.nufft_2d(x, y, self.im_size)


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: GoldenAngleArgs, z_dim_fft=True
):
    kspace_raw_data = raw_data * recon_args.amplitude_scale_factor
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajectory(
        kspace_raw_data.shape[2], recon_args.spoke_len
    )
    if recon_args.partial_fourier_flag:
        kspace_data_centralized, kspace_data_mask = comp.centralize_kspace(
            kspace_data=kspace_raw_data,
            acquire_length=recon_args.partition_num,
            center_idx_in_acquire_lenth=recon_args.kspace_centre_partition_num,
            full_length=recon_args.slice_num,
            dim=1,
        )
    else:
        kspace_data_centralized = kspace_raw_data
        kspace_data_mask = None
    if z_dim_fft:
        # kspace_data_z = comp.batch_process(
        #     batch_size=1, device=recon_args.device
        # )(comp.ifft_1D)(kspace_data_centralized, dim=1, norm="backward")
        kspace_data_z = comp.ifft_1D(
            kspace_data_centralized, dim=1, norm="ortho"
        )
        return dict(
            kspace_data_centralized=kspace_data_centralized,
            kspace_data_z=kspace_data_z,
            kspace_traj=kspace_traj,
            kspace_data_mask=kspace_data_mask,
        )
    else:
        return dict(
            kspace_data_centralized=kspace_data_centralized,
            kspace_traj=kspace_traj,
            kspace_data_mask=kspace_data_mask,
        )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: GoldenAngleArgs,
    return_multi_channel: bool = False,
    density_compensation_func: Callable = voronoi_density_compensation,
    *args,
    **kwargs,
):
    kspace_data_centralized, kspace_data_z, kspace_traj = (
        data_preprocessed["kspace_data_centralized"],
        data_preprocessed["kspace_data_z"],
        data_preprocessed["kspace_traj"],
    )

    csm = get_csm_lowk_xy(
        kspace_data_centralized,
        kspace_traj,
        recon_args.im_size,
        0.05,
    )

    kspace_density_compensation = density_compensation_func(
        kspace_traj,
        device=kspace_traj.device,
    )
    kspace_data = comp.radial_spokes_to_kspace_point(
        kspace_data_z * kspace_density_compensation
    )
    kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)

    img_multi_ch = comp.nufft_adj_2d(
        kspace_data,
        kspace_traj,
        recon_args.im_size,
    )
    img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    return img

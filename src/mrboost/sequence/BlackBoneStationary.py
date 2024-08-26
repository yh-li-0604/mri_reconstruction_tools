# %%
from dataclasses import dataclass, field
from typing import Dict

import einx
import torch

# from icecream import ic
from plum import dispatch

from mrboost import computation as comp
from mrboost.coil_sensitivity_estimation import (
    get_csm_lowk_xyz,
)
from mrboost.density_compensation import (
    ramp_density_compensation,
)

# from mrboost.io_utils import *
from mrboost.sequence.GoldenAngle import GoldenAngleArgs


@dataclass
class BlackBoneStationaryArgs(GoldenAngleArgs):
    # start_spokes_to_discard: int = field(init=False)  # to reach the steady state
    nSubVolSlc: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.nSubVolSlc = round(self.slice_num * 3 / 4)


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: BlackBoneStationaryArgs
):
    _dict = preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
        raw_data, recon_args
    )
    kspace_data_centralized, kspace_data_z, kspace_traj = (
        _dict["kspace_data_centralized"],
        _dict["kspace_data_z"],
        _dict["kspace_traj"],
    )
    csm = get_csm_lowk_xyz(
        kspace_data_centralized,
        kspace_traj,
        recon_args.im_size,
        [0.03, 0.03],
        # recon_args.device,
    )

    return dict(
        kspace_data_z=kspace_data_z,
        kspace_traj=kspace_traj,
        csm=csm,
    )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: BlackBoneStationaryArgs,
    return_multi_channel: bool = False,
    *args,
    **kwargs,
):
    kspace_data_z, kspace_traj, csm = (
        data_preprocessed["kspace_data_z"],
        data_preprocessed["kspace_traj"],
        data_preprocessed["csm"],
    )
    kspace_data_z = comp.radial_spokes_to_kspace_point(kspace_data_z)
    kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)
    kspace_density_compensation = ramp_density_compensation(
        kspace_traj,
        im_size=recon_args.im_size,
    )
    img_multi_ch = comp.nufft_adj_2d(
        kspace_data_z * kspace_density_compensation,
        kspace_traj,
        recon_args.im_size,
    )
    img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    return img

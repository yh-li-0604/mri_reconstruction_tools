import os
import numpy as np
from mrboost import reconstruction as recon
from dlboost.utils.tensor_utils import complex_normalize_abs_95
from mrboost.sequence.CAPTURE_VarW_NQM_DCE_PostInj import (
    CAPTURE_VarW_NQM_DCE_PostInj_Args,
    mcnufft_reconstruct,
    preprocess_raw_data,
)
from mrboost.io_utils import get_raw_data
from mrboost.computation import normalization
import torch

def recon_one_scan_P2P(
    raw_data, shape_dict, mdh, twixobj, phase_num=5, time_per_contrast=10
):
    # raw_data, shape_dict, mdh, twixobj = get_raw_data(dat_file_to_recon)
    args = CAPTURE_VarW_NQM_DCE_PostInj_Args(
        shape_dict,
        mdh,
        twixobj,
        phase_num=5, # 10
        time_per_contrast=10, # 20
        frequency_encoding_oversampling_removed=True,
        device=torch.device("cuda:1"),
    )
    data_dict_func = preprocess_raw_data(raw_data, args)
    images, csm = mcnufft_reconstruct(data_dict_func, args)
    mean, std = complex_normalize_abs_95(
        images, expand=False
    )
    images_normed = images / std
    return (
        data_dict_func["kspace_data_z"],
        data_dict_func["kspace_traj"],
        images_normed,
        csm,
        mean,
        std,
    )

# Load your data
raw_data, shape_dict, mdh, twixobj = get_raw_data('/data/anlab/RawData_MR/CCIR_01168_ONC-DCE/ONC-DCE-014/meas_MID00099_FID12331_CAPTURE_FA14_5_Dyn.dat')

# Run the reconstruction
_, _, output, _, _, _ = recon_one_scan_P2P(raw_data, shape_dict, mdh, twixobj)

# Save the output
np.save(os.path.join("/data/anlab/Yunhe/mri_reconstruction_tools/output/01", "images.npy"), output)

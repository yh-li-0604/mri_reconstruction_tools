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

RAW_DATA_PATH="/data/anlab/RawData_MR/CCIR_1194/meas_MID00091_FID17947_CAPTURE_Cardiac/meas_MID00091_FID17947_CAPTURE_Cardiac.dat"
raw_data, shape_dict, mdh, twixobj = get_raw_data(RAW_DATA_PATH)

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
# import os
from pathlib import Path

# from typing import Dict
import fire
import torch

# from icecream import ic
from scipy.io import savemat

from mrboost import reconstruction as recon
from mrboost.io_utils import check_mk_dirs


def DynGAVibe(data_dir, output_path, contra_num=20, spokes_per_contra=320):
    # %%
    raw_data, shape_dict, mdh, twixobj = recon.get_raw_data(data_dir)

    # %%
    ga_args = recon.DynGoldenAngleVibeArgs(
        shape_dict,
        mdh,
        twixobj,
        contra_num=contra_num,
        spokes_per_contra=spokes_per_contra,
    )

    # %%
    preprocessed_data = recon.preprocess_raw_data(raw_data, ga_args)

    # %%
    results = recon.mcnufft_reconstruct(preprocessed_data, ga_args, False)
    output_path = Path(output_path)

    check_mk_dirs(str(output_path.parent))
    if output_path.suffix == ".mat":
        savemat(
            str(output_path),
            {"image": results.abs().numpy(force=True)},
        )
    elif output_path.suffix == ".pt":
        torch.save(results, str(output_path))


def BlackBoneStationary(data_dir, output_path):
    # %%
    raw_data, shape_dict, mdh, twixobj = recon.get_raw_data(data_dir)
    ga_args = recon.BlackBoneStationaryArgs(
        shape_dict,
        mdh,
        twixobj,
    )
    preprocessed_data = recon.preprocess_raw_data(raw_data, ga_args)
    results = recon.mcnufft_reconstruct(preprocessed_data, ga_args, False)

    output_path = Path(output_path)
    check_mk_dirs(str(output_path.parent))
    if output_path.suffix == ".mat":
        savemat(
            str(output_path),
            {"image": results.abs().numpy(force=True)},
        )
    elif output_path.suffix == ".pt":
        torch.save(results, str(output_path))
        
if __name__ == "__main__":
    fire.Fire()

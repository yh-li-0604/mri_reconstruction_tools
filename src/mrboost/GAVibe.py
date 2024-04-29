# %%

# from icecream import ic
from scipy.io import savemat

# %%
from mrboost import reconstruction as recon

# %%
raw_data, shape_dict, mdh, twixobj = recon.get_raw_data(
    "/bmrc-an-data/Chunwei/WIP/Test_DCE_SoS/RawData/Subject_1/meas_MID00260_FID13077_fl3d_vibe_GA_15_6600.dat"
)

# %%
ga_args = recon.DynGoldenAngleVibeArgs(
    shape_dict, mdh, twixobj, contra_num=20, spokes_per_contra=320
)

# %%
preprocessed_data = recon.preprocess_raw_data(raw_data, ga_args)

# %%
results = recon.mcnufft_reconstruct.invoke(recon.DynGoldenAngleVibeArgs)(
    preprocessed_data
)

savemat(
    "/bmrc-an-data/Chunxu/MR_Recon/meas_MID00260_FID13077_fl3d_vibe_GA_15_6600.mat",
    {"image": results.abs().numpy(force=True)},
)

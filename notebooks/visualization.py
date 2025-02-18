from matplotlib import pyplot as plt
from torch.fft import fft, fftshift, ifft, ifftshift
import torch 
import os
import numpy as np
import torch
from torch.fft import fft, fftshift
import scipy.signal
from dlboost.utils.tensor_utils import complex_normalize_abs_95
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Callable, Dict
import einx
import numpy as np
import torch
from plum import dispatch
from mrboost.coil_sensitivity_estimation import get_csm_lowk_xyz
from mrboost.density_compensation import ramp_density_compensation
from mrboost.sequence.CAPTURE_VarW_NQM_DCE_PostInj import (
    CAPTURE_VarW_NQM_DCE_PostInj_Args,
    mcnufft_reconstruct,
)
from mrboost import computation as comp


def mcnufft_reconstruct_csm_check(
    data_preprocessed: Dict[str, torch.Tensor],
    raw_data,
    recon_args: CAPTURE_VarW_NQM_DCE_PostInj_Args,  # Adapt recon_args as needed
    return_multi_channel: bool = False,
    density_compensation_func: Callable = ramp_density_compensation,
    csm_xy_z_lowk_ratio=[0.05, 0.05],
    *args,
    **kwargs,
):
    # Load the k-space data and trajectory
    kspace_raw_data = (
        raw_data[:, 1:, recon_args.start_spokes_to_discard :, :]
        * recon_args.amplitude_scale_factor
    )
    
    kspace_raw_data = (
        raw_data[:, 1:, recon_args.start_spokes_to_discard :, :]
        * recon_args.amplitude_scale_factor
    )
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajectory(
        raw_data.shape[2], recon_args.spoke_len
    )[:, recon_args.start_spokes_to_discard :]
    kspace_data_centralized, kspace_data_mask = comp.centralize_kspace(
        kspace_data=kspace_raw_data,
        acquire_length=recon_args.partition_num,
        center_idx_in_acquire_lenth=recon_args.kspace_centre_partition_num - 1,
        # -1 because of navigator, and this number is index started from 0
        full_length=recon_args.slice_num,
        dim=1,
    )

    kspace_data_z = comp.ifft_1D(kspace_data_centralized, dim=1, norm="ortho")
    
    # Compute coil sensitivity map (CSM)
    csm = get_csm_lowk_xyz(
        data_preprocessed["kspace_data_csm"],
        data_preprocessed["kspace_traj_csm"],
        recon_args.im_size,  # Adjust im_size as needed
        [0.05, 0.05],
    )

    # Perform density compensation
    _kspace_density_compensation = density_compensation_func(kspace_traj)

    # Convert radial spokes to k-space data
    _kspace_data = comp.radial_spokes_to_kspace_point(
        kspace_data_z * _kspace_density_compensation
    )
    _kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)

    # Apply Non-uniform Fast Fourier Transform (NUFFT)
    img_multi_ch = comp.nufft_adj_2d(
        _kspace_data,
        _kspace_traj,
        recon_args.im_size,
    )

    # Combine the channels using coil sensitivity map (CSM)
    img = einx.sum("[ch] d w h", img_multi_ch * csm.conj())
    img_weighted = img_multi_ch * csm.conj()

    return img.cpu(), csm, img_weighted.cpu()

def process_and_plot(data_dict_func, args, coil_num, angle, output_path, start_idx=100, end_idx=300):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    r = data_dict_func["time_curve_total"]
    cardiac_tcurve = r[:, coil_num, angle].cpu().numpy()

    plt.plot(cardiac_tcurve[start_idx:end_idx])
    plt.savefig(os.path.join(output_path, "cardiac_time_curve.png"))
    plt.close()

    R = data_dict_func["R_total"].cpu().numpy()
    cardiac_fcurve = R[:, coil_num, angle]
    
    mu = data_dict_func["mu"]
    
    # print(mu)
    lower_cutoff = np.maximum(mu - 0.31, 0.6)

    Fs = args.Fs
    b = scipy.signal.firwin(
        20, [lower_cutoff / (Fs / 2), (mu + 0.31) / (Fs / 2)], window="hamming", pass_zero=False
    )
    a = 1

    r_max_low_pass = scipy.signal.filtfilt(b, a, cardiac_tcurve)
    r_max_SG = scipy.signal.filtfilt(
        b, a, scipy.signal.savgol_filter(cardiac_tcurve, 5, 1)
    )
    r_max_filtered = r_max_low_pass.copy()
    r_max_filtered[0:10], r_max_filtered[-10:] = r_max_SG[0:10], r_max_SG[-10:]

    plt.plot(r_max_filtered[start_idx:end_idx])
    plt.ylim
    plt.savefig(os.path.join(output_path, "filtered_time_curve.png"))
    plt.close()

    f = np.linspace(-Fs/2, Fs/2, r_max_filtered.shape[0])
    R_filtered = fft(torch.tensor(r_max_filtered))

    plt.plot(f, abs(fftshift(R_filtered)))
    plt.xlim([0, 4])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity")
    plt.title("Filtered Cardiac Curve in Frequency Space")
    plt.savefig(os.path.join(output_path, "filtered_frequency_curve.png"))
    plt.close()

    plt.plot(f, cardiac_fcurve)
    plt.xlim([0, 4])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity")
    plt.title("Cardiac Curve in Frequency Space")
    plt.savefig(os.path.join(output_path, "cardiac_frequency_curve.png"))
    plt.close()

    proj = data_dict_func["proj"]

    rp = phase_rotation(proj.cpu().numpy(), angle, Fs)

    w = data_dict_func["rotated_proj"]
    w = w.cpu().numpy()
    proj_coil = rp[:, :, coil_num]

    plt.figure(figsize=(10, 6))
    original_mean = np.mean(w[start_idx:end_idx, coil_num, angle])
    filtered_mean = np.mean(r_max_filtered[start_idx:end_idx])
    offset = original_mean - filtered_mean

    plt.imshow(abs(proj_coil[:, start_idx:end_idx]), aspect='auto', origin='lower')
    x_vals = np.linspace(start_idx, end_idx, end_idx - start_idx)
    plt.plot(r_max_filtered[start_idx:end_idx] + offset, 'r-', linewidth=2)
    plt.savefig(os.path.join(output_path, "overlay_plot.png"))
    plt.close()

def phase_rotation(projections, angle, Fs):
    N = projections.shape[1]

    f = np.linspace(-0.5 * Fs, 0.5 * Fs - Fs / N, num=N)
    phase_rotation_factors = np.exp(
        -1j * 2 * np.pi * np.arange(1, 361) / 360
    )

    r = np.empty((projections.shape[0], projections.shape[1], projections.shape[2]))
    r = (phase_rotation_factors[angle] * projections).real

    return r

def csm_check(
    data_dict_func, 
    raw_data,
    args,
    output_path, 
    axial_idx=163,
    sagittal_idx=29,
    coronal_idx=177,
):
    imax = data_dict_func["m"]
    coil_num = data_dict_func["i"]
    
    img, csm, img_weighted = mcnufft_reconstruct_csm_check(data_dict_func, raw_data, args)
    mean, std = complex_normalize_abs_95(
        img, expand=False
    )
    images_normed = img / std
    
    csm = csm.cpu().numpy()
    coils = csm.shape[0]
    
    if coils == 42:
        l, w = 6, 7
    elif coils == 48:
        l, w = 6, 8
    
    fig, axs = plt.subplots(w, l, figsize=(20, 20))
    for i in range(coils):
        ax = axs[i//6, i%6]

        masked_image = abs(images_normed[:, axial_idx, :].T) * abs(csm[i, :, axial_idx, :].T)
        ax.imshow(masked_image, cmap='gray', aspect='auto')  # 关键修改
        ax.set_title(f"Coil {i}", fontweight='bold' if i == coil_num else 'normal')
        ax.axis("off")
    plt.savefig(os.path.join(output_path, f"axial_csm_{coil_num}.png"))
    plt.close()

    # sagittal切面
    fig, axs = plt.subplots(w, l, figsize=(20, 20))
    for i in range(coils):
        ax = axs[i//6, i%6]
        masked_image = abs(images_normed[sagittal_idx, :, :]) * abs(csm[i, sagittal_idx, :, :])
        ax.imshow(masked_image, cmap='gray', aspect='auto')  # 关键修改
        ax.set_title(f"Coil {i}", fontweight='bold' if i == coil_num else 'normal')
        ax.axis("off")
    plt.savefig(os.path.join(output_path, f"sagittal_csm_{coil_num}.png"))
    plt.close()

    # coronal切面
    fig, axs = plt.subplots(w, l, figsize=(20, 20))
    for i in range(coils):
        ax = axs[i//6, i%6]

        masked_image = abs(images_normed[:, :, coronal_idx].T) * abs(csm[i, :, :, coronal_idx].T)
        ax.imshow(masked_image, cmap='gray', aspect='auto')  # 关键修改
        ax.set_title(f"Coil {i}", fontweight='bold' if i == coil_num else 'normal')
        ax.axis("off")
    plt.savefig(os.path.join(output_path, f"coronal_csm_{coil_num}.png"))
    plt.close()
    
    # fig, axs = plt.subplots(w, l, figsize=(20, 20))
    # for i in range(coils):
    #     ax = axs[i//6, i %6]
    #     ax.imshow(abs(images_normed[:, axial_idx, :].T), cmap='gray', aspect='auto')
    #     ax.imshow(abs(csm[i, :, axial_idx, :].T), alpha=0.5, aspect='auto')
    #     if i == coil_num:
    #         ax.set_title(f"Coil {i}", fontweight='bold')
    #     else:
    #         ax.set_title(f"Coil {i}")
    #     ax.axis("off")
    # plt.savefig(os.path.join(output_path, f"axial_csm_{coil_num}.png"))
    
    # fig, axs = plt.subplots(w, l, figsize=(20, 20))
    # for i in range(coils):
    #     ax = axs[i//6, i %6]
    #     ax.imshow(abs(images_normed[sagittal_idx, :, :]), cmap='gray', aspect='auto')
    #     ax.imshow(abs(csm[i, sagittal_idx, :, :]), alpha=0.5, aspect='auto')
    #     if i == coil_num:
    #         ax.set_title(f"Coil {i}", fontweight='bold')
    #     else:
    #         ax.set_title(f"Coil {i}")
    #     ax.axis("off")
    # plt.savefig(os.path.join(output_path, f"sagittal_csm_{coil_num}.png"))
    
    # fig, axs = plt.subplots(w, l, figsize=(20, 20))
    # for i in range(coils):
    #     ax = axs[i//6, i %6]
    #     ax.imshow(abs(images_normed[:, :, coronal_idx].T), cmap='gray', aspect='auto')
    #     ax.imshow(abs(csm[i, :, :, coronal_idx].T), alpha=0.5, aspect='auto')
    #     if i == coil_num:
    #         ax.set_title(f"Coil {i}", fontweight='bold')
    #     else:
    #         ax.set_title(f"Coil {i}")
    #     ax.axis("off")
    # plt.savefig(os.path.join(output_path, f"coronal_csm_{coil_num}.png"))
    


def create_animation(image, id, save_path, slice, slice_index):
    combined_data = np.stack(image, axis=0)
    combined_data = np.transpose(combined_data, (1, 2, 3, 0))
    combined_data = np.abs(combined_data)
    
    num_volumes = combined_data.shape[-1]
    type_ = ["respiratory", "cardiac", "combined"][id]
    # save_path = f"/data/anlab/Yunhe/mri_reconstruction_tools/output/{type_}"
    vmin = 95.20
    vmax = 83111.20
    t = False

    match slice:
        case "coronal":
            shape = combined_data[:, :, slice_index, :].shape
            combined_slice = np.zeros((shape[1], shape[0], shape[2]))
            slice_img = np.fliplr(combined_data[:, :, slice_index, 0].T)
            for i in range(combined_data.shape[-1]):
                combined_slice[:, :, i] = np.fliplr(combined_data[:, :, slice_index, i].T)
            aspect = 1.125/3
        case "sagittal":
            slice_img = combined_data[slice_index, :, :, 0]
            combined_slice = combined_data[slice_index, :, :, :]
            aspect = 1.125/1.125
        case "transverse":
            t = True
            slice_img = combined_data[:, slice_index, :, 0].T
            combined_slice = combined_data[:, slice_index, :, :]
            aspect = 1.125/3
        case _:
            raise ValueError(f"Invalid slice type: {slice}")

    fig, ax = plt.subplots()
    img = ax.imshow(slice_img, cmap='gray', animated=True, aspect=aspect, vmin=vmin, vmax=vmax)
    ax.set_title(f"{type_}_{slice} Animation")
    ax.axis('off')

    def update(frame):
        if t:
            video_frame = combined_slice[:, :, frame].T
        else:
            video_frame = combined_slice[:, :, frame]
        img.set_array(video_frame)
        return img,

    ani = FuncAnimation(fig, update, frames=num_volumes, interval=500, blit=True)
    plt.show()
    ani.save(f'{save_path}/{slice}_slices_animation_{slice_index}.mp4', writer='ffmpeg', fps=5)
    
# Example usage:
# process_and_plot(data_dict_func, args, coil_num, angle, "/path/to/output")
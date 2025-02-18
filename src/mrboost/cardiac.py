import os
import numpy as np
from mrboost import reconstruction as recon
from dlboost.utils.tensor_utils import complex_normalize_abs_95
from mrboost.sequence.CAPTURE_VarW_NQM_DCE_PostInj import (
    CAPTURE_VarW_NQM_DCE_PostInj_Args,
    mcnufft_reconstruct,
)
from mrboost.io_utils import get_raw_data
from mrboost.computation import normalization
import torch
from numbers import Number
from types import NoneType
from typing import Sequence
from mrboost.sequence.CAPTURE_VarW_NQM_DCE_PostInj import (
    mcnufft_reconstruct,
)
import einx

# import numpy as np
import scipy
import torch
import torch.nn.functional as F

# from juliacall import Main as jl
# jl.include("/data-local/anlab/Chunxu/mri_reconstruction_tools/src/mrboost/computation.jl")
from einops import rearrange, reduce, repeat

# from icecream import ic
from jax import numpy as np
from jaxtyping import Shaped
from plum import dispatch, overload
from pytorch_finufft.functional import (
    FinufftType1,
    FinufftType2,
    finufft_type1,
    finufft_type2,
)
from torch.fft import fft, fftshift, ifft, ifftshift
from tqdm import tqdm

from mrboost.io_utils import *
from mrboost import computation as comp

from mrboost.type_utils import (
    ComplexImage2D,
    KspaceData,
    KspaceSpokesData,
    KspaceSpokesTraj,
    KspaceTraj,
)

from dataclasses import dataclass, field
from typing import Callable, Dict
import einx
import numpy as np
import torch
from plum import dispatch
from mrboost.coil_sensitivity_estimation import get_csm_lowk_xyz
from mrboost.density_compensation import ramp_density_compensation
# from visualization import process_and_plot, csm_check, create_animation, create_animation_by_phase


def median_filter_1d(R, kernel_size=5):
    """
    Apply a median filter of size `kernel_size` along the first dimension of R.
    R has shape [N, C1, C2], and we want to filter along the N dimension.
    
    In this case:
    N = 2290 (frequency dimension)
    C1 = 42  (coils)
    C2 = 360 (angles)

    We'll do:
    1) Reshape R to (1, C1*C2, N) so we have a single batch and multiple channels.
    2) Pad along the last dimension.
    3) Unfold to extract windows of size `kernel_size`.
    4) Take median along the kernel dimension.
    5) Reshape back to (N, C1, C2).
    """
    N, C1, C2 = R.shape
    pad = kernel_size // 2

    # Move frequency dimension (N) to last to simulate a sequence dimension
    # and combine C1 and C2 into a single channel dimension:
    # From [N, C1, C2] to [1, C1*C2, N]
    R_reshaped = R.permute(1, 2, 0).contiguous().view(1, C1 * C2, N)

    # Pad along the sequence dimension (last dimension)
    R_padded = F.pad(R_reshaped, (pad, pad), mode='reflect') 
    # Now shape is [1, C1*C2, N+2*pad]

    # Use unfold to extract sliding windows of size kernel_size along the last dimension
    # unfold for 1D: input shape [N, C, L], we get windows [N, C, L_out, kernel_size]
    windows = R_padded.unfold(dimension=2, size=kernel_size, step=1)  
    # windows shape: [1, C1*C2, N, kernel_size]

    # Take median along the kernel dimension (last dimension)
    R_filtered_reshaped = windows.median(dim=-1)[0]  # [1, C1*C2, N]

    # Reshape back to [N, C1, C2]
    R_filtered = R_filtered_reshaped.view(C1, C2, N).permute(2, 0, 1).contiguous()
    return R_filtered


def gaussian_kernel(size=51, sigma=55.0):
    # Generate a 1D Gaussian kernel
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    return kernel

def gaussian_convolution(filtered_R, kernel_size=51, sigma=5.0):
    """
    Convolve the [N, C1, C2] frequency-domain data with a 1D Gaussian kernel along the N dimension.
    
    Arguments:
        filtered_R (torch.Tensor): Input tensor of shape [N, C1, C2].
        kernel_size (int): Length of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: Convolved result with shape [N, C1, C2].
    """

    # Extract dimensions
    N, C1, C2 = filtered_R.shape
    # Create Gaussian kernel
    g_kernel = gaussian_kernel(size=kernel_size, sigma=sigma).to(filtered_R.device)
    # Reshape data to [1, C1*C2, N]
    data_reshaped = filtered_R.permute(1, 2, 0).contiguous().view(1, C1*C2, N)
    # Reshape kernel to [out_channels, in_channels/groups, kernel_size]
    # We'll use groups = C1*C2, so each channel is convolved independently.
    g_kernel = g_kernel.view(1, 1, -1)
    g_kernel = g_kernel.repeat(C1*C2, 1, 1)

    # Perform convolution with groups = C1*C2
    conv_output = F.conv1d(
        data_reshaped, 
        g_kernel, 
        padding=kernel_size // 2,
        groups=C1*C2
    )
    
    # Reshape back to [N, C1, C2]
    conv_output = conv_output.view(C1, C2, N).permute(2, 0, 1).contiguous()
    
    return conv_output

def peak_quality(R, f, low_freq_range=(0.8, 1.8)):
    """
    Vectorized computation of a simple peak quality metric for each coil-angle pair in R.

    Args:
        R (torch.Tensor): Frequency-domain data of shape [N, C1, C2].
        f (torch.Tensor): Frequencies of shape [N].
        low_freq_range (tuple): The (low, high) frequency range in which to look for peaks.

    Returns:
        torch.Tensor: A score matrix of shape [C1, C2] indicating peak quality for each coil-angle pair.
    """
    # Create a mask for the low frequency range
    low_mask = (torch.abs(f) >= low_freq_range[0]) & (torch.abs(f) <= low_freq_range[1])
    R_low_freq = R[low_mask, :, :]  # shape: [N_low, C1, C2]
    N_low = R_low_freq.shape[0]

    if N_low < 3:
        # Not enough points to detect peaks
        return torch.zeros(R.shape[1], R.shape[2], device=R.device)
    
    # Identify local maxima:
    # For an index i to be a peak:
    # R_low_freq[i] > R_low_freq[i-1] and R_low_freq[i] > R_low_freq[i+1]
    # We'll compare shifted versions of R_low_freq:
    R_prev = R_low_freq[:-2, :, :]  # shifted forward by 1
    R_mid = R_low_freq[1:-1, :, :]  # middle
    R_next = R_low_freq[2:, :, :]   # shifted backward by 1

    # Boolean mask of peaks:
    is_peak = (R_mid > R_prev) & (R_mid > R_next)
    # is_peak shape: [N_low-2, C1, C2]

    # Replace non-peaks with -inf so max will ignore them:
    # We'll pad is_peak back to N_low by considering only the middle portion.
    peak_values = torch.where(is_peak, R_mid, torch.tensor(float('-inf'), device=R.device))

    # peak_values shape: [N_low-2, C1, C2]

    # Find the max peak value for each coil-angle pair
    max_peak_values, max_peak_indices = torch.max(peak_values, dim=0)  # shape: [C1, C2]

    # For plateau calculation, we take the mean of all points (in the low freq range) except the peak:
    # sum_all: sum over the low freq dimension
    sum_all = R_low_freq.sum(dim=0)  # shape: [C1, C2]

    # Number of points in low freq range is N_low
    # Subtract the max peak value and divide by (N_low - 1) to get the average of non-peak points
    # If no valid peak was found (all -inf), max_peak_values might be -inf. Handle that:
    no_peak_mask = torch.isinf(max_peak_values)
    max_peak_values[no_peak_mask] = 0.0  # Replace -inf with 0 for arithmetic

    plateau_value = (sum_all - max_peak_values) / (N_low - 1)
    plateau_value[no_peak_mask] = sum_all[no_peak_mask] / N_low  # If no peak, plateau is just the mean

    # Compute a simple score: peak_value / plateau_value
    # Higher is better: strong peak compared to surrounding plateau.
    score = max_peak_values / (plateau_value + 1e-9)

    # If no peak was found, score might be meaningless:
    score[no_peak_mask] = 0.0

    return score


def arrange_array(sorted_idx, label):
    Y = np.zeros_like(label)
    for i, idx in enumerate(sorted_idx):
        Y[idx] = label[i]
    return Y

def hamming_filter(nonzero_width_percent: float, width: int) -> np.ndarray:
    nonzero_width = round(width * nonzero_width_percent)
    pad_width_L = round((width - nonzero_width) // 2)
    pad_width_R = width - nonzero_width - pad_width_L
    hamming_weights = np.float32(np.hamming(nonzero_width))
    W = np.pad(hamming_weights, pad_width=(pad_width_L, pad_width_R))
    return W

def tuned_and_robust_estimation_cardiac(
    navigator: np.ndarray,
    percentW: float,
    Fs,
    FOV,
    ndata,
    device=torch.device("cuda"),
):
    """
    return channel and rotation index and generated curve
    """
    col_num, line_num, ch_num = navigator.shape

    # To reduce noise, the navigator k-space data were apodized using a Hamming window.
    W = hamming_filter(percentW / 100, col_num)
    W = repeat(
        W,
        "col_num -> col_num line_num ch_num",
        line_num=line_num,
        ch_num=ch_num,
    )

    # New quality metric block begin
    N = navigator.shape[1]
    f = torch.linspace(-0.5 * Fs, 0.5 * Fs - Fs / N, steps=N, device=device)
    # compute the ifft of weighted navigator, using the representation in CAPTURE paper
    # col_num->x, line_num->n, ch_num->i, tuning_num->m
    
    # low_pass_mask = (torch.abs(f) <= 3).float()
    # low_pass_mask = low_pass_mask.view(1, N, 1).repeat(col_num, 1, ch_num)

    # Apply the low-pass filter to the weighted navigator
    # K_weighted = torch.from_numpy(navigator).to(f.device) * low_pass_mask
    K_weighted = torch.from_numpy(W * navigator).to(f.device)
    
    projections = fftshift(
        ifft(ifftshift(K_weighted, dim=0), dim=0), dim=0
    )  # shape is x n i
    
    # print(projections.shape)

    # shape is m=100
    phase_rotation_factors = torch.exp(
        -1j * 2 * torch.pi * torch.arange(1, 361, device=f.device) / 360
    )
    r = torch.empty(
        (projections.shape[1], projections.shape[2], 360), device=f.device
    )
    
    window_width = 5
    half_window = window_width // 2
    x_dim, y_dim, z_dim = projections.shape
    sliding_window = torch.ones((window_width,), device=f.device) / window_width
    padding = 0 if window_width == 1 else half_window

    # # Apply sliding window mean first
    # projections_reshaped = projections.permute(1, 2, 0).reshape(-1, 1, x_dim)  # (2290*42, 1, 320)

    # # Perform 1D convolution for sliding window mean
    # projections_smoothed_reshaped = torch.nn.functional.conv1d(
    #     projections_reshaped,
    #     sliding_window.view(1, 1, -1),  # Kernel: (1, 1, window_width)
    #     padding=padding
    # )
    # Separate real and imaginary parts
    projections_real = projections.real
    projections_imag = projections.imag

    # Reshape for convolution
    projections_real_reshaped = projections_real.permute(1, 2, 0).reshape(-1, 1, x_dim)
    projections_imag_reshaped = projections_imag.permute(1, 2, 0).reshape(-1, 1, x_dim)

    # Perform sliding window convolution for both real and imaginary parts
    projections_smoothed_real_reshaped = torch.nn.functional.conv1d(
        projections_real_reshaped,
        sliding_window.view(1, 1, -1),
        padding=padding,
    )
    projections_smoothed_imag_reshaped = torch.nn.functional.conv1d(
        projections_imag_reshaped,
        sliding_window.view(1, 1, -1),
        padding=padding,
    )

    # Reshape back to original dimensions
    projections_smoothed_real = projections_smoothed_real_reshaped.reshape(y_dim, z_dim, x_dim).permute(2, 0, 1)
    projections_smoothed_imag = projections_smoothed_imag_reshaped.reshape(y_dim, z_dim, x_dim).permute(2, 0, 1)

    # Combine real and imaginary parts into a complex tensor
    projections_smoothed = projections_smoothed_real + 1j * projections_smoothed_imag

    # Apply phase rotation to the smoothed data
    for m in range(360):
        # Apply phase rotation
        rotated_projections = (phase_rotation_factors[m] * projections_smoothed).real  # (320, 2290, 42)
    
        # Find the index of the max value along the x-dimension
        r[:, :, m] = torch.argmax(rotated_projections, dim=0)  # (2290, 42)
    
    w = r.clone()
    
    # A = torch.einsum('xni,m->xnim',projections,phase_rotation_factors).real # np.multiply.outer(projections, phase_rorate..)
    # r = torch.argmax(A,dim=0).to(torch.double)+1 # 'x n i m -> n i m'
    R = torch.abs(
        fftshift(fft(r - reduce(r, "n i m -> i m", "mean"), dim=0), dim=0)
    )

    # Need to be changed 
    # lowfreq_integral = reduce(
    #     R[(torch.abs(f) < 1.7) * (torch.abs(f) > 1.0)], "f i m -> i m", "sum"
    # )
    # highfreq_integral = reduce(R[(torch.abs(f) < 1.0) + (torch.abs(f) > 2.5)], "f i m -> i m", "sum")
    # r_range = reduce(r, "n i m -> i m", "max") - reduce(
    #     r, "n i m -> i m", "min"
    # )
    # lower_bound = torch.full_like(r_range, 10 / (FOV / (ndata / 2)))
    # # what does this FOV/ndata use for
    # determinator = torch.maximum(r_range, lower_bound)
    # Q = lowfreq_integral / highfreq_integral / determinator
    # Q_np = Q.numpy(force=True)  # faster than matlab version 10x
    Q, mu_f = quality(R, r, f, FOV, ndata)
    sigma = 0.156
    Q_np = Q.numpy(force=True)
    
    i_max, m_max = np.unravel_index(np.argmax(Q_np), Q_np.shape)
    
    # i_max = 27
    # m_max = 314
    
    # projection_max = projections[:, :, i_max]
    r_max = r[:, i_max, m_max].numpy(force=True)
    # new quality metric block end
    mu = r_max.mean()
    mu_f = mu_f[i_max, m_max].numpy(force=True)
    # filter high frequency signal
    b = scipy.signal.firwin(
        12, [(mu_f - 2*sigma) / (Fs / 2), (mu_f + 2*sigma) / (Fs / 2)], window="hamming", pass_zero=False
    )
    a = 1
    r_max_low_pass = scipy.signal.filtfilt(b, a, r_max)
    r_max_SG = scipy.signal.filtfilt(
        b, a, scipy.signal.savgol_filter(r_max, 5, 1)
    )
    r_max_filtered = r_max_low_pass.copy()
    r_max_filtered[0:10], r_max_filtered[-10:] = r_max_SG[0:10], r_max_SG[-10:]
    r_max_filtered += mu
    
    return i_max, m_max, torch.from_numpy(r_max_filtered), R[:, i_max, m_max], Q_np, R, r, projections_smoothed, w, mu_f

def assign_combined_phases(
    spoke_count,
    sorted_idx,
    phase_num,
):
    bin_size = spoke_count // phase_num
    # respiratory_bin_size = spoke_count // respiratory_phase_num

    phases = torch.arange(phase_num).repeat_interleave(bin_size)
    # respiratory_phases = torch.arange(respiratory_phase_num).repeat_interleave(respiratory_bin_size)

    phase_mapping = arrange_array(sorted_idx, phases)
    # respiratory_phase_mapping = arrange_array(respiratory_sorted_idx, respiratory_phases)

    # combined_phase_indices = (
    #     respiratory_phase_mapping * cardiac_phase_num + cardiac_phase_mapping
    # )
    # return combined_phase_indices
    return phase_mapping

def bin_second_phases(
    data, # should be a different curve, torch.Size[2170], if first bin is res, then this should be cardiac
    ph_idx,
    cardiac_phase_num,
    respiratory_phase_num,
):
    combined_phase_mapping = np.zeros(data.shape[0], dtype=np.int64)
    for phase_idx in range(respiratory_phase_num):
        mask = ph_idx == phase_idx
        # Select the spokes corresponding to this phase
        selected_spokes = data[mask]  
        
        _, cardiac_phase_idx = torch.sort(selected_spokes)
        cardiac_phase_mapping = assign_combined_phases(
            cardiac_phase_idx.shape[0],
            cardiac_phase_idx,
            cardiac_phase_num,
        )
        combined_phase_mapping[mask] = phase_idx * cardiac_phase_num + cardiac_phase_mapping
        
    return combined_phase_mapping


def bin_data_to_phases(
    data,
    ph_idx,
    cardiac_phase_num,
    respiratory_phase_num,
):
    *leading_dims, spoke_num, spoke_len = data.shape
    total_phases = cardiac_phase_num * respiratory_phase_num
    binned_data_list = [None] * total_phases
    for phase_idx in range(total_phases):
        # Mask to select the spokes for this phase
        mask = ph_idx == phase_idx
        # Select the spokes corresponding to this phase
        selected_spokes = data[..., mask, :]  
        # Store the selected data
        binned_data_list[phase_idx] = selected_spokes
    
    spokes_per_bin = binned_data_list[0].shape[-2]
    
    binned_data = torch.zeros(
        (respiratory_phase_num, cardiac_phase_num, *leading_dims, spokes_per_bin, spoke_len),
        dtype=data.dtype,
        device=data.device,
    )
    
    for phase_idx in range(total_phases):
        resp_phase = phase_idx // cardiac_phase_num
        card_phase = phase_idx % cardiac_phase_num
        spokes = binned_data_list[phase_idx]
        if spokes is not None:
            num_spokes = spokes.shape[-2]
            binned_data[resp_phase, card_phase, ..., :num_spokes, :] = spokes
        
    return binned_data
    

def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: CAPTURE_VarW_NQM_DCE_PostInj_Args
):
    nav = (
        einx.rearrange(
            "ch_num spoke_num spoke_len -> spoke_len spoke_num ch_num",
            raw_data[:, 0, recon_args.start_spokes_to_discard :, :],
        )
        * recon_args.amplitude_scale_factor
    )
    # ch = 1
    # Manually set the cardiac curve, currently it is the auto matched manual only in this case
    ch_c, rotation_c, cardiac_curve, r, q, Rm, t, projections, w, mu = tuned_and_robust_estimation_cardiac(
        navigator=nav.numpy(),
        percentW=recon_args.percentW,
        # percentW=5,
        Fs=recon_args.Fs,
        FOV=recon_args.FOV,
        ndata=recon_args.spoke_len,
        device=recon_args.device,
    )
    
    ch_r, rotation_r, respiratory_curve = comp.tuned_and_robust_estimation(
        navigator=nav.numpy(),
        percentW=recon_args.percentW,
        Fs=recon_args.Fs,
        FOV=recon_args.FOV,
        ndata=recon_args.spoke_len,
        device=recon_args.device,
    )
    
    cardiac_curve = cardiac_curve[
        recon_args.binning_start_idx : recon_args.binning_end_idx
    ]
    respiratory_curve = respiratory_curve[
        recon_args.binning_start_idx : recon_args.binning_end_idx
    ]

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
    
    spoke_count = recon_args.binning_end_idx - recon_args.binning_start_idx
    cardiac_phase_num = recon_args.phase_num
    respiratory_phase_num = 5

    # _, cardiac_sorted_idx = torch.sort(cardiac_curve)  
    _, respiratory_sorted_idx = torch.sort(respiratory_curve)  

    res_phase_indices = assign_combined_phases(
        spoke_count,
        respiratory_sorted_idx,
        respiratory_phase_num,
    )
    
    combined_phase_indices = bin_second_phases(
        cardiac_curve,
        res_phase_indices,
        cardiac_phase_num=cardiac_phase_num,
        respiratory_phase_num=respiratory_phase_num
    )
    
    

    # Bin the data
    (
        _kspace_traj,
        _kspace_data_z,
        _kspace_data_centralized,
        _kspace_data_mask,
    ) = map(
        bin_data_to_phases,
        [
            kspace_traj[:, recon_args.binning_start_idx : recon_args.binning_end_idx],
            kspace_data_z[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx, :
            ],
            kspace_data_centralized[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx, :
            ],
            kspace_data_mask[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
            ],
        ],
        [combined_phase_indices] * 4,
        [cardiac_phase_num] * 4,
        [respiratory_phase_num] * 4,
    )


    return {
        "r": r,
        "Quality_metric": q,
        "R_total": Rm,
        "filtered_curve": cardiac_curve,
        "i": ch_c,
        "m": rotation_c,
        "time_curve_total": t,
        "proj": projections, 
        "rotated_proj": w,
        "r_curve":respiratory_curve,
        "kspace_traj": _kspace_traj,
        "kspace_data_z": _kspace_data_z,
        "kspace_data_centralized": _kspace_data_centralized,
        "kspace_data_mask": _kspace_data_mask,
        "kspace_data_csm": kspace_data_centralized[
            :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
        ],
        "kspace_traj_csm": kspace_traj[
            :, recon_args.binning_start_idx : recon_args.binning_end_idx
        ],
        "mu": mu,
    }


def quality(R, r, f, FOV, ndata):
    R = torch.tensor(R)
    # R = normalize_f(R)
    R_filtered = median_filter_1d(R, kernel_size=3)
    R_convolved = gaussian_convolution(R_filtered, kernel_size=51, sigma=5)


    mu, score = peak_detection(R_convolved, f, low_freq_range=(0.8, 1.5))
    sigma = 0.156 # 80% of the energy is preserved in +- 0.2 Hz, for example, 0.8 - 1.3 Hz for peak @ 1.1 Hz
    l = torch.max(mu, torch.full_like(mu, 0.6)) - 2 * sigma
    u = mu + 2 * sigma

    f_abs = torch.abs(f).unsqueeze(-1).unsqueeze(-1)  # shape: [N,1,1]
    mask_low = (f_abs < u) & (f_abs > l)  # shape: [N, C1, C2]
    mask_high = (f_abs < l) + (f_abs > u)  # shape: [N, C1, C2]

    # Apply mask to R_c and then reduce
    # R_c[(mask)] won't directly work in einops with different shapes; instead do elementwise:
    low_signal = (R_filtered) * mask_low.float()
    high_signal = (R) * mask_high.float()

    lowfreq_integral = reduce(low_signal, "f i m -> i m", "sum")
    highfreq_integral = reduce(high_signal, "f i m -> i m", "sum")

    # FOV = args.FOV
    # ndata = args.spoke_len
    r_range = torch.tensor(reduce(r, "n i m -> i m", "max") - reduce(
        r, "n i m -> i m", "min"
        ))
    lower_bound = torch.full_like(r_range, 10 / (FOV / (ndata / 2)))

    determinator = torch.maximum(r_range, lower_bound)
    Q = score * lowfreq_integral / highfreq_integral / torch.sqrt(determinator)
    # Q = score * lowfreq_integral / highfreq_integral
    
    # print(torch.max(lowfreq_integral), torch.max(highfreq_integral))
    return Q,  mu

def normalize_f(spectrum, eps=1e-9):
    """
    Normalize the input spectrum based on its total energy.
    
    Args:
        spectrum (torch.Tensor or np.ndarray): Input spectrum (1D or multi-dimensional).
        eps (float): A small constant to prevent division by zero.
    
    Returns:
        normalized (torch.Tensor): The normalized spectrum, where the total energy = 1.
    """
    # Convert input to torch.Tensor if it isn't already
    if not isinstance(spectrum, torch.Tensor):
        spectrum = torch.from_numpy(spectrum)
    
    # 1) Remove the mean (optional, to ensure no DC offset)
    min_val = spectrum.min()
    centered = spectrum - min_val

    # 2) Normalize by total energy
    total_energy = (centered**2).sum().sqrt()  # L2 norm
    normalized = centered / (total_energy + eps)
    
    return normalized


def peak_detection(R, f, low_freq_range=(0.5, 2.0)):
    """
    Compute a peak quality metric for each coil-angle pair, ensuring:
    1) The main peak is within (0.5, 2.0) Hz.
    2) Identify ±3σ region around the max peak frequency (mu = peak freq).
    3) Within ±3σ, ensure no significant second peak exists.
    4) If a second peak is found in that region, penalize the score.

    Args:
        R (torch.Tensor): Frequency-domain data, shape [N, C1, C2].
        f (torch.Tensor): Frequencies, shape [N].
        low_freq_range (tuple): The (low, high) frequency range for initial peak detection.
        second_peak_ratio_threshold (float): Threshold for considering a second peak significant.
    
    Returns:
        mu (torch.Tensor): [C1, C2], frequencies of the identified main peaks.
    """
    # Step 1: Select frequencies in the low_freq_range
    low_mask = (torch.abs(f) >= low_freq_range[0]) & (torch.abs(f) <= low_freq_range[1])
    R_low_freq = R[low_mask, :, :]  # shape: [N_low, C1, C2]
    f_low_freq = torch.abs(f[low_mask])
    N_low = R_low_freq.shape[0]

    # If not enough points, return zeros
    if N_low < 3:
        C1, C2 = R.shape[1], R.shape[2]
        return torch.zeros(C1, C2, device=R.device), torch.zeros(C1, C2, device=R.device), torch.zeros(C1, C2, device=R.device)

    # Step 2: Identify local maxima (peaks)
    R_prev = R_low_freq[:-2, :, :]
    R_mid = R_low_freq[1:-1, :, :]
    R_next = R_low_freq[2:, :, :]

    is_peak = (R_mid > R_prev) & (R_mid > R_next)
    # peak_values align with R_mid, which corresponds to f_low_freq[1:-1]
    peak_values = torch.where(is_peak, R_mid, torch.tensor(float('-inf'), device=R.device))

    # Find max peak value and index for each coil-angle
    max_peak_values, max_indices = torch.max(peak_values, dim=0)  # [C1, C2]

    no_peak_mask = torch.isinf(max_peak_values)
    max_peak_values[no_peak_mask] = 0.0

    # Compute plateau value
    sum_all = R_low_freq.sum(dim=0)  # [C1, C2]
    plateau_value = (sum_all - max_peak_values) / (N_low - 1)
    plateau_value[no_peak_mask] = sum_all[no_peak_mask] / N_low

    # Basic score
    score = max_peak_values / (plateau_value + 1e-9)
    score[no_peak_mask] = 0.0

    # Step 3: Compute mu and sigma for each pair
    # max_indices are indices in [0, N_low-2], corresponding to R_mid.
    # The actual frequency index for the peak in f_low_freq is max_indices+1
    peak_freq_idx = (max_indices + 1)  # shift by one because R_mid corresponds to f_low_freq[1:-1]
    # handle no peaks
    peak_freq_idx[no_peak_mask] = 0

    # mu = frequency at the peak
    mu = f_low_freq[peak_freq_idx]  # [C1, C2]
    
    return mu, score


# @dispatch
# def mcnufft_reconstruct(
#     data_preprocessed: Dict[str, torch.Tensor],
#     recon_args: CAPTURE_VarW_NQM_DCE_PostInj_Args,
#     return_multi_channel: bool = False,
#     density_compensation_func: Callable = ramp_density_compensation,
#     csm_xy_z_lowk_ratio=[0.05, 0.05],
#     *args,
#     **kwargs,
# ):
#     # Extract preprocessed k-space data, trajectories, and masks
#     kspace_data_centralized, kspace_data_z, kspace_traj, kspace_mask = (
#         data_preprocessed["kspace_data_centralized"],
#         data_preprocessed["kspace_data_z"],
#         data_preprocessed["kspace_traj"],
#         data_preprocessed["kspace_data_mask"], 
#     )
    
#     print(kspace_data_z.shape)
#     print(kspace_traj.shape)

#     # Compute coil sensitivity maps (CSMs)
#     csm = get_csm_lowk_xyz(
#         data_preprocessed["kspace_data_csm"],
#         data_preprocessed["kspace_traj_csm"],
#         recon_args.im_size,
#         csm_xy_z_lowk_ratio,
#     )

#     # Initialize list to store reconstructed images
#     images = []

#     # Iterate over cardiac and respiratory phases
#     cardiac_phase_num = 35
#     respiratory_phase_num = 5
#     phases = []
#     for ph in range(cardiac_phase_num):

#         print(f"reconstructing  phase {ph+1}")
#         _kspace_density_compensation = density_compensation_func(
#             kspace_traj[ph],
#             energy_match_radial_with_cartisian=True
#             # device=kspace_traj.device,
#         )
#         _kspace_data = comp.radial_spokes_to_kspace_point(
#             kspace_data_z[ph] * _kspace_density_compensation
#         )
#         _kspace_traj = comp.radial_spokes_to_kspace_point(
#             kspace_traj[ph]
#         )
#         img_multi_ch = comp.nufft_adj_2d(
#             _kspace_data,
#             _kspace_traj,
#             recon_args.im_size,
#         )
#         img = einx.sum("[ch] d w h", img_multi_ch * csm.conj())
#         phases.append(img.cpu())
#         # images.append(torch.stack(phases, dim=0))       
        
#     # Stack cardiac phases
#     return torch.stack(phases, dim=0), csm




if __name__=="__main__":
    list_file_path = "/data/anlab/Yunhe/mri_reconstruction_tools/data_path/cardiac_file.list"
    output_path = "/data/anlab/Yunhe/mri_reconstruction_tools/output/03.3"

    # Open the list file and iterate through each line
    with open(list_file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace (e.g., newlines)
            file_path = line.strip()
            name = os.path.splitext(os.path.basename(file_path))[0]
            # Check if the line is not empty
            
            if file_path:
                # print(name)
                output = os.path.join(output_path, name)
                if not os.path.exists(output):
                    os.makedirs(output)
                    
                raw_data, shape_dict, mdh, twixobj = get_raw_data(file_path)
                args = CAPTURE_VarW_NQM_DCE_PostInj_Args(
                    shape_dict,
                    mdh,
                    twixobj,
                    phase_num=5, # 10
                    time_per_contrast=88, # 20
                    # frequency_encoding_oversampling_removed=True,
                    device=torch.device("cuda:0"),
                )
                Fs = args.Fs
                data_dict_func = preprocess_raw_data(raw_data, args)
                coil_num = data_dict_func["i"]
                rotation_angle = data_dict_func["m"]
                print(coil_num, rotation_angle)
                
                # process_and_plot(data_dict_func, args, coil_num, rotation_angle, output)
                # csm_check(
                #     data_dict_func,
                #     raw_data,
                #     args,
                #     output,
                # )
                
                image, csm = mcnufft_reconstruct(data_dict_func, args)
                # mean, std = complex_normalize_abs_95(
                #     image, expand=False
                # )
                # images_normed = image / std
                image = image.numpy(force=True)
                print(image.shape)
                
                # for id in range(3):
                #     slice = ["sagittal", "coronal", "transverse"]
                #     idx = [29, 177, 163]
                #     for i in range(3):
                #         # create_animation(
                #         #     image, 
                #         #     id, 
                #         #     output, 
                #         #     slice[i],
                #         #     idx[i]
                #         # )
                #         create_animation_by_phase(
                #             image, 
                #             id, 
                #             output, 
                #             slice[i],
                #             idx[i]
                #         )
                #         create_animation_by_phase(
                #             image, 
                #             id, 
                #             output, 
                #             slice[i],
                #             idx[i],
                #             phases=35
                #         )
                        
                #     plt.close("all")
                
            
            
from itertools import product
from dataclasses import dataclass, field
from typing import Callable, Optional, Type,Any
from torch import TensorType
import numpy as np
import torch
import einops as eo
import torchkbnufft as tkbn
from mapvbvd.mapVBVD import myAttrDict,evalMDH
from src.io_utils import *
from src.twix_metadata_def import *
import src.computation as comp
import src.preprocessing as pre
from src.coil_sensitivity_estimation import CoilSensitivityEstimator, Lowk_2D_CSE, lowk_xy, lowk_xyz


@dataclass
class CAPTURE_VarW_NQM_DCE_PostInj_Args:
    """Class for keeping track of useful arguments for reconstruction CAPTURE_VarW_NQM_DCE_PostInj."""
    twixobj: myAttrDict
    mdh: Any
    shape_dict: dict
    partition_num: int =field(init=False)
    slice_num: int=field(init=False)
    ch_num: int=field(init=False)
    kspace_centre_partition_num: int =field(init=False)
    start_spokes_to_discard: int=field(init=False)
    spoke_num: int=field(init=False)
    spoke_len: int=field(init=False)
    contra_num: int=field(init=False)
    spokes_per_contra: int=field(init=False)
    spokes_per_phase: int=field(init=False)
    spokes_to_skip: int=field(init=False)
    FOV: float =field(init=False)
    TR: float =field(init=False)
    T: float =field(init=False)
    Fs: float=field(init=False)
    im_size: tuple=field(init=False)
    grid_size: tuple =field(init=False)
    sorted_r_idx: torch.Tensor =field(init=False)
    # datFileLocation: Path = field(default_factory=Path)
    first_slice: int = 0
    last_slice: int = 80
    phase_num: int = 5
    injection_time: int = 30
    duration_to_reconstruct: int = 340
    time_per_contrast: int = 10
    percentW: float = 12.5
    cache_folder: Path = Path('.')/'cache'
    device: torch.device = torch.device('cuda')

    def __post_init__(self):
        self.slice_num = round(self.twixobj.hdr.Meas.lImagesPerSlab*(1+self.twixobj.hdr.Meas.dSliceOversamplingForDialog))
        self.kspace_centre_partition_num = int(self.mdh.ushKSpaceCentrePartitionNo[0]) # important! -1 because of nav

        self.ch_num = self.shape_dict['ch_num']
        self.total_partition_num = self.shape_dict['partition_num']
        self.partition_num = self.total_partition_num-1 # this number does not contain navigator
        self.spoke_num = self.shape_dict['spoke_num']
        self.spoke_len = self.shape_dict['spoke_len']
        
        assert self.last_slice <= self.slice_num, f"Try to set {self.last_slice=} <= {self.slice_num=}"
        assert self.first_slice < self.slice_num, f"Try to set {self.first_slice=} < {self.slice_num=}"
        assert self.last_slice > self.first_slice, f"Try to set {self.last_slice=} > {self.first_slice=}"
        

        self.TR = self.twixobj.hdr.MeasYaps[('alTR', '0')]/1000
        self.T = self.TR*self.total_partition_num*1e-3+18.86e-3  # 19e-3 is for Q-fat sat
        self.Fs = 1/self.T
        self.FOV = self.twixobj.hdr.Meas.RoFOV# type: ignore
        
        # TODO what this means???
        self.start_spokes_to_discard = max(max(self.phase_num, 10), self.phase_num*np.ceil(10/self.phase_num))

        nSpokesToWorkWith = np.floor(self.duration_to_reconstruct/self.T)
        # per contrast from injection time. (Jan )
        # Exclude the first 10 prep spokes
        self.spokes_to_skip = round(
            (self.injection_time-self.time_per_contrast)/self.T)  # self.start_idx
        nSpokesPerContrast = np.floor(self.time_per_contrast/self.T)
        self.spokes_per_contra = int(
            nSpokesPerContrast-np.mod(nSpokesPerContrast, self.phase_num))
        self.contra_num = int(np.floor(nSpokesToWorkWith/self.spokes_per_contra))
        self.spokes_per_phase = int(nSpokesPerContrast/self.phase_num)

        # self.im_size = (self.spoke_len//2, self.spoke_len//2)
        self.im_size = (self.spoke_len, self.spoke_len)
        self.grid_size = (int(2*self.spoke_len), int(2*self.spoke_len))

        


def nufft_CAPTURE_VarW_NQM_DCE_PostInj_Args_init(args: CAPTURE_VarW_NQM_DCE_PostInj_Args, registered_vars, scan_meta_data_list, shape_dict):
    # ch_num = shape_dict['ch_num']
    args.spoke_num = int(registered_vars['lRadialViews'])
    args.partition_num = int(registered_vars['NParMeas'])
    numberOfSlices = int(registered_vars['lImagesPerSlab'])
    sliceOverSampleing = registered_vars[(
        'sKSpace', 'dSliceOversamplingForDialog')]
    args.slice_num = round(numberOfSlices*(1+sliceOverSampleing))
    args.kspace_centre_partition_num = scan_meta_data_list[-1]['kspace_centre_partition_num']

    # next we check the validity of input first slice and last slice
    assert args.last_slice <= args.slice_num, f"Try to set {args.last_slice=} <= {numberOfSlices=}"
    assert args.first_slice < args.slice_num, f"Try to set {args.first_slice=} < {numberOfSlices=}"
    assert args.last_slice > args.first_slice, f"Try to set {args.last_slice=} > {args.first_slice=}"

    args.FOV = registered_vars['RoFOV']
    # TODO what is this used for?
    args.spoke_len = int(2*registered_vars['NImageLins'])
    # LOW_BOUND = 30/(FOV/(ndata/2))

    # iou.search_for_keywords_in_AAA(registered_vars,'SliceThickness')
    # TODO didn't find this argument
    # sliceThickness = 1/numberOfSlices*registered_vars['SliceThickness']
    isSAG = registered_vars['lSag']
    isCor = registered_vars['lCor']
    isTra = (not isSAG) and (not isCor)

    # Read TR from raw data
    # iou.search_for_keywords_in_AAA(registered_vars,'alTR')
    args.TR = registered_vars[('alTR', '0')]/1000
    args.T = args.TR*args.partition_num*1e-3+18.86e-3  # 19e-3 is for Q-fat sat
    args.Fs = 1/args.T

    # TODO what this means???
    # args.start_idx = max(max(args.phase_num, 10), args.phase_num*np.ceil(10/args.phase_num))

    nSpokesToWorkWith = np.floor(args.duration_to_reconstruct/args.T)
    # per contrast from injection time. (Jan )
    # Exclude the first 10 prep spokes
    args.spokes_to_skip = round(
        (args.injection_time-args.time_per_contrast)/args.T)  # args.start_idx
    nSpokesPerContrast = np.floor(args.time_per_contrast/args.T)
    args.spokes_per_contra = int(
        nSpokesPerContrast-np.mod(nSpokesPerContrast, args.phase_num))
    args.contra_num = int(np.floor(nSpokesToWorkWith/args.spokes_per_contra))
    args.spokes_per_phase = int(nSpokesPerContrast/args.phase_num)

    # args.grid_size = (args.spoke_len, args.spoke_len)
    args.im_size = (args.spoke_len//2, args.spoke_len//2)
    # args.im_size = (args.spoke_len, args.spoke_len)
    args.ch_num = shape_dict['ch_num']

    return args

    # get channel number
    """
    s1 = 'dRawDataCorrectionFactorIm';
    ind1 = strfind(AAA',s1);
    ind1 = ind1(1);
    s2 = 'Method."ComputeScan">';
    ind2 = strfind(AAA',s2);
    ind2 = ind2(find(ind2>ind1,1));
    textOfInterest = AAA(ind1+200:ind2)';
    indQuote = strfind(textOfInterest,'"');
    indQuote = reshape(indQuote,[2,length(indQuote)/2]);
    clear coilIDs
    for k = 1:size(indQuote,2)
        coilIDs{k} = char(textOfInterest(indQuote(2*(k-1)+1)+1:indQuote(2*k)-1));
    end
    disp(coilIDs)
    numberOfChannels = length(coilIDs); 
    """
    # TODO this matlab method reads out 39 channels, but I find 42 channel in AAA


def nufft_CAPTURE_VarW_NQM_DCE_PostInj(args: CAPTURE_VarW_NQM_DCE_PostInj_Args, CoilSensitivityEstimator):
    registered_vars, current_pos = pre.read_protocol(
        datFileLocation=args.datFileLocation, which_scan=-1)

    shape_dict = dict(
        spoke_num=int(registered_vars['iNoOfFourierLines']),
        partition_num=int(registered_vars['lPartitions']),
        echo_num=int(registered_vars['NEcoMeas']))

    # first read all mdh header
    scan_meta_data_list, shape_dict = pre.read_scan_meta_data(
        args.datFileLocation, current_pos, shape_dict)
    # now shape dict have spoke_num, partition_num, echo_num, ch_num, spoke_len

    args = nufft_CAPTURE_VarW_NQM_DCE_PostInj_Args_init(
        args, registered_vars, scan_meta_data_list, shape_dict)

    nav, kspace_data = pre.read_navigator_kspace_data(
        args.datFileLocation, scan_meta_data_list, shape_dict)

    # spoke_len, spoke_num, ch_num
    nav = np.ascontiguousarray(nav[:, args.start_idx:args.spoke_num, :])

    ch, rotation, respiratory_curve = comp.tuned_and_robust_estimation(
        navigator=nav, percentW=args.percentW, Fs=args.Fs, FOV=args.FOV, ndata=args.spoke_len)
    # separate the respiratory curve into different time periods (contrasts)
    respiratory_curve_contrast = eo.rearrange(
        respiratory_curve[args.spokes_to_skip:args.spokes_to_skip +
                          args.contra_num*args.spokes_per_contra],
        '(contra spokes_per_contra) -> contra spokes_per_contra',
        contra=args.contra_num,
        spokes_per_contra=args.spokes_per_contra,
    )
    # in each of the contrast, we sort the respiratory curve in order to classify respiratory phases
    sorted_r, args.sorted_r_idx = torch.sort(
        respiratory_curve_contrast, dim=-1)

    # Wait for steady-state [make sure phase_num>=10]
    # everytime when partition=0, there is a navigator
    kspace_data_selected = torch.from_numpy(
        eo.rearrange(
            kspace_data[:, :, 1:args.partition_num, :, 0],
            'spoke_len spoke_num partition_num ch_num -> ch_num partition_num spoke_num spoke_len'
        )
    ).narrow(dim=-2, start=args.spokes_to_skip, length=args.contra_num*args.spokes_per_contra)
    kspace_data_selected = kspace_data_selected/kspace_data_selected.abs().max()
    kspace_data_centralized = comp.centralize_kspace(
        kspace_data=kspace_data_selected,
        acquire_length=args.partition_num,
        full_length=args.slice_num,
        center_in_acquire_lenth=args.kspace_centre_partition_num,
        dim=1)
    kspace_data_z = comp.batch_process(batch_size=2, device=args.device)(
        comp.ifft_1D)(kspace_data_centralized, dim=1)
    kspace_data_for_nufft = torch.flip(kspace_data_z, dims=(1,))[
        :, args.first_slice:args.last_slice]

    # shape of kspace_traj: complex_num_ch spoke_num spoke_len
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajctory(
        args.spoke_num, args.spoke_len
    ).narrow(dim=-2, start=args.spokes_to_skip, length=args.contra_num*args.spokes_per_contra)
    # build nufft operators
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=args.im_size).to(
        args.device)  # , grid_size=grid_size)

    # DCE reconstruction
    # k_space_density_compensation, it is flattened due to nufft, now recover it
    kspace_density_compensation = eo.rearrange(
        tkbn.calc_density_compensation_function(
            ktraj=eo.rearrange(
                kspace_traj, 'c spoke_num spoke_len -> c (spoke_num spoke_len)'),
            im_size=args.im_size),
        'ch b (spokes_num spoke_len) -> spokes_num spoke_len ',
        b=1,
        ch=1,
        spokes_num=args.contra_num*args.spokes_per_contra,
        spoke_len=args.spoke_len)

    kspace_traj, k_density_compensation, kspace_data = map(
        comp.data_binning,
        [kspace_traj, kspace_density_compensation, kspace_data_for_nufft],
        [args.sorted_r_idx]*3, [args.contra_num]*3,
        [args.spokes_per_contra]*3, [args.phase_num]*3,
        [args.spokes_per_phase]*3)

    cse = CoilSensitivityEstimator(
        kspace_data_centralized, kspace_traj, adjnufft_ob, batch_size=2, device=torch.device('cpu'))

    img = torch.zeros((args.contra_num, args.phase_num, args.last_slice -
                      args.first_slice, args.im_size[0], args.im_size[1]), dtype=torch.complex64)
    for t, ph in product(range(args.contra_num), range(args.phase_num)):
        print('NUFFT for contrast:{}, phase:{}'.format(t, ph))
        sensitivity_map = cse[t, ph][:, args.first_slice:args.last_slice]
        img[t, ph, :, :, :] = \
            eo.reduce(
                comp.batch_process(batch_size=2, device=args.device)(comp.recon_adjnufft)(
                    k_density_compensation*kspace_data[t, ph],
                    sensitivity_map,
                    kspace_traj=kspace_traj[t, ph],
                    adjnufft_ob=adjnufft_ob), 'slice ch w h -> slice w h', 'sum')
    print('MCNUFFT reconstruction finished')
    img = (img-img.mean())/img.std()
    return img

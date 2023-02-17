from itertools import islice, product
from typing import Callable, Optional, Sequence, Type,Any,Union
import numpy as np
import torch
import einops as eo
import torchkbnufft as tkbn
from src.mapvbvd.mapVBVD import mapVBVD
from src.io_utils import *
from src.twix_metadata_def import *
import src.computation as comp
import src.preprocessing as pre
from src.coil_sensitivity_estimation import CoilSensitivityEstimator, Lowk_2D_CSE, lowk_xy, lowk_xyz
from src.density_compensation import cihat_pipe_density_compensation, voronoi_density_compensation


class Reconstructor():
    def __init__(self,
        dat_file_location: Path = Path('/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE/ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat'),
        which_slice: Union[int, Sequence] = (0,80),
        device: torch.device = torch.device('cpu'),
        *args, **kwargs) -> None:
        self.dat_file_location = dat_file_location
        self.which_slice = which_slice
        self.device = device
        # data_raw = self.__get_raw_data(dat_file_location)
        # self.__args_init(*args,**kwargs)
        # data_dict = self.__data_preprocess(data_raw)
        # self.img = self.reconstruction(data_dict)

    def get_raw_data(self,dat_file_location):
        twixobj, self.mdh = mapVBVD(dat_file_location)
        self.twixobj = twixobj[-1]
        self.twixobj.image.squeeze=True
        self.twixobj.image.flagRemoveOS = False
        data_raw = eo.rearrange(self.twixobj.image[''],'spoke_len ch_num spoke_num partition_num -> ch_num partition_num spoke_num spoke_len')
        self.shape_dict = eo.parse_shape(data_raw, 'ch_num partition_num spoke_num spoke_len')
        return data_raw
    
    def __args_init(self,*args,**kwargs):
        self.amplitude_scale_factor =  80 * 20 * 131072 / 65536 * 20000

        self.slice_num = round(self.twixobj.hdr.Meas.lImagesPerSlab*(1+self.twixobj.hdr.Meas.dSliceOversamplingForDialog))
        if self.which_slice ==-1:
            self.which_slice = (0,self.slice_num)
        self.which_slice = (self.which_slice,self.which_slice+1) if isinstance(self.which_slice,int) else self.which_slice
        assert self.which_slice[1] <= self.slice_num, f"Try to set {self.which_slice[1]=} <= {self.slice_num=}"
        self.slice_to_recon = [i for i in range(self.slice_num)][slice(*self.which_slice)]
        self.kspace_centre_partition_num = int(self.mdh.ushKSpaceCentrePartitionNo[0]) # important! -1 because of nav

        self.ch_num = self.shape_dict['ch_num']
        self.total_partition_num = self.shape_dict['partition_num']
        self.partition_num = self.total_partition_num-1 # this number does not contain navigator
        self.spoke_num = self.shape_dict['spoke_num']
        self.spoke_len = self.shape_dict['spoke_len']
        
        self.TR = self.twixobj.hdr.MeasYaps[('alTR', '0')]/1000
        self.T = self.TR*self.total_partition_num*1e-3+18.86e-3  # 19e-3 is for Q-fat sat
        self.Fs = 1/self.T
        self.FOV = self.twixobj.hdr.Meas.RoFOV# type: ignore
        
        self.im_size = (self.spoke_len, self.spoke_len)
        self.grid_size = (int(2*self.spoke_len), int(2*self.spoke_len))
        
    def args_init_post(self,*args,**kwargs):
        pass

    def args_init_before(self,*args,**kwargs):
        pass
    
    def args_init(self,*args,**kwargs):
        self.args_init_before(*args,**kwargs)
        self.__args_init(*args,**kwargs)
        self.args_init_post(*args, **kwargs)
        
    def data_preprocess(self,data_raw): 
        # need to overload in child reconsturctor
        kspace_traj = None
        return dict(kspace_data = data_raw, kspace_traj = kspace_traj)
    
    def reconstruction(self, data_dict):
        kspace_data, kspace_traj = data_dict['kspace_data'],data_dict['kspace_traj']

class CAPTURE_VarW_NQM_DCE_PostInj(Reconstructor):
    def __init__(self,
        dat_file_location: Path = Path('/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE/ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat'),
        phase_num: int = 5,
        which_slice: Union[int, Sequence] = (0,80),
        which_contra: Union[int, Sequence] = (0,34),
        which_phase: Union[int, Sequence] = (0,5),
        injection_time: int = 30,
        duration_to_reconstruct: int = 340,
        time_per_contrast: int = 10,
        percentW: float = 12.5,
        cache_folder: Path = Path('.')/'cache',
        device: torch.device = torch.device('cuda')) -> None:
        self.phase_num = phase_num
        self.which_contra = which_contra
        self.which_phase = which_phase
        self.injection_time = injection_time
        self.duration_to_reconstruct= duration_to_reconstruct
        self.time_per_contrast= time_per_contrast
        self.percentW= percentW
        self.cache_folder= cache_folder
        super().__init__(dat_file_location, which_slice, device)

    def args_init_before(self,*args,**kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def args_init_post(self):
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

        self.binning_start_idx = self.spokes_to_skip -self.start_spokes_to_discard
        self.binning_end_idx = self.spokes_to_skip-self.start_spokes_to_discard+self.contra_num*self.spokes_per_contra

        if self.which_contra ==-1:
            self.which_contra = (0,self.contra_num)
        if self.which_phase ==-1:
            self.which_phase = (0,self.phase_num)
        self.which_contra, self.which_phase = map(lambda x: (x,x+1) if isinstance(x,int) else x,[self.which_contra,self.which_phase])
        self.contra_to_recon = [i for i in range(self.contra_num)][slice(*self.which_contra)]
        self.phase_to_recon = [i for i in range(self.phase_num)][slice(*self.which_phase)] 

        # build nufft operators
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=self.im_size,grid_size=self.grid_size).to(
            self.device)
        self.nufft_ob = tkbn.KbNufft(im_size=self.im_size,grid_size=self.grid_size).to(
            self.device)

    def data_preprocess(self, data_raw):
        data_raw *= self.amplitude_scale_factor
        kspace_data_raw = data_raw[:,1:,self.start_spokes_to_discard:,:]
        nav = eo.rearrange(data_raw[:,0,self.start_spokes_to_discard:,:], 'ch_num spoke_num spoke_len -> spoke_len spoke_num ch_num')
        kspace_traj =2*torch.pi* comp.generate_golden_angle_radial_spokes_kspace_trajctory(
            self.spoke_num, self.spoke_len)[:,self.start_spokes_to_discard:]
        
        sorted_r_idx = self.navigator_preprocess(nav)
        kspace_data_centralized, kspace_data_z = self.kspace_data_preprocess(kspace_data_raw)
        
        cse = Lowk_2D_CSE(
            # kspace_data_, kspace_traj_, adjnufft_ob, batch_size=2, device=self.device)
            kspace_data_centralized, kspace_traj,self.nufft_ob, self.adjnufft_ob,hamming_filter_ratio=0.05, batch_size=1, device=self.device)

        kspace_traj,  kspace_data = map(
            comp.data_binning,
            [kspace_traj[:,self.binning_start_idx:self.binning_end_idx],
            kspace_data_z[:,:,self.binning_start_idx:self.binning_end_idx]],
            [sorted_r_idx]*2, [self.contra_num]*2,
            [self.spokes_per_contra]*2, [self.phase_num]*2,
            [self.spokes_per_phase]*2)
        return dict(kspace_data=kspace_data,kspace_traj=kspace_traj,cse=cse)

    def navigator_preprocess(self,nav):
        ch, rotation, respiratory_curve = comp.tuned_and_robust_estimation(
            navigator=nav, percentW=self.percentW, Fs=self.Fs, FOV=self.FOV, ndata=self.spoke_len)
        # here rotation is index of 100 different degree, to get same with cihat, please+1
        respiratory_curve_contrast = eo.rearrange(
            respiratory_curve[self.binning_start_idx:self.binning_end_idx],
            '(contra spokes_per_contra) -> contra spokes_per_contra',
            contra=self.contra_num,
            spokes_per_contra=self.spokes_per_contra)
        # separate the respiratory curve into different time periods (contrasts)

        sorted_r, sorted_r_idx = torch.sort(
            respiratory_curve_contrast, dim=-1)
        # in each of the contrast, we sort the respiratory curve in order to classify respiratory phases
        return sorted_r_idx
    
    def kspace_data_preprocess(self, kspace_data_raw):
        kspace_data_centralized = comp.centralize_kspace(
            kspace_data=torch.from_numpy(kspace_data_raw),
            acquire_length=self.partition_num,
            # -1 because of navigator, and this number is index started from 0
            center_idx_in_acquire_lenth=self.kspace_centre_partition_num-1,
            full_length=self.slice_num,
            dim=1)

        kspace_data_z = comp.batch_process(batch_size=2, device=self.device)(
            comp.ifft_1D)(kspace_data_centralized, dim=1)
        # flip or not, doesnt change much
        kspace_data_z = torch.flip(kspace_data_z, dims=(1,))
        return kspace_data_centralized, kspace_data_z
    
    def reconstruction(self,data_dict):
        kspace_data, kspace_traj, cse = data_dict['kspace_data'],data_dict['kspace_traj'],data_dict['cse']
        # if isinstance(self.which_contra,int):

        img = torch.zeros((len(self.contra_to_recon), len(self.phase_to_recon), len(self.slice_to_recon),self.im_size[0]//2, self.im_size[1]//2), dtype=torch.complex64)
        for t, ph in product(self.contra_to_recon, self.phase_to_recon):
            print('NUFFT for contrast:{}, phase:{}'.format(t, ph))
            sensitivity_map = cse[t, ph].to(torch.complex64).conj()[:, self.slice_to_recon]
            output = comp.batch_process(batch_size=2,device = self.device)(comp.recon_adjnufft)(
                        kspace_data[t, ph, :],#, self.first_slice:args.last_slice],
                        kspace_traj=kspace_traj[t,ph],
                        adjnufft_ob=self.adjnufft_ob,
                        density_compensation_func = voronoi_density_compensation
                        )[:,self.slice_to_recon,self.im_size[0]//2-self.im_size[0]//4:self.im_size[0]//2+self.im_size[0]//4,
                          self.im_size[1]//2-self.im_size[1]//4:self.im_size[1]//2+self.im_size[1]//4]
            output *= sensitivity_map
            img[t, ph, :, :, :] = \
                eo.reduce(
                    torch.flip( output, (-1,) ), 'ch slice w h -> slice w h', 'sum')
        return img

    def forward(self):
        data_raw = self.get_raw_data(self.dat_file_location)
        self.args_init()
        data_dict = self.data_preprocess(data_raw)
        img = self.reconstruction(data_dict)
        return img

if __name__=="__main__":
    # if you want to reconstruct step by step (cache each of step in memory if in jupyter interative mode)
    reconstructor = CAPTURE_VarW_NQM_DCE_PostInj(dat_file_location=Path('/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE/ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat'), which_slice=50,which_contra=0,which_phase=0)
    data_raw = reconstructor.get_raw_data(reconstructor.dat_file_location)
    reconstructor.args_init()
    data_dict = reconstructor.data_preprocess(data_raw)
    img = reconstructor.reconstruction(data_dict)

    # If you want just want to reconstract
    reconstructor = CAPTURE_VarW_NQM_DCE_PostInj(dat_file_location=Path('/data/anlab/PET_MOCO/PETMRdata/CAPTURE_DCE/ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat'), which_slice=50,which_contra=0,which_phase=0)
    img = reconstructor.forward()
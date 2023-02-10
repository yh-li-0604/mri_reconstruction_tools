from dataclasses import field
from typing import Optional, Sequence
import numpy as np
import torch
import einops as eo
import torchkbnufft as tkbn
from pipe import Pipe
from scipy.spatial import Voronoi
from src import computation as comp
from toolz.functoolz import curry, pipe


def voronoi_density_compensation(kspace_traj: torch.Tensor,
                                 im_size: Sequence[int],
                                 grid_size: Optional[Sequence[int]] = None):
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    # print(spoke_shape)
    device = kspace_traj.device
    kspace_traj_complex = eo.rearrange(
            kspace_traj, 'c spokes_num spoke_len -> (spokes_num spoke_len) c')
    kspace_traj_complex = torch.view_as_complex(kspace_traj_complex.contiguous()).to(device)
    kspace_traj_complex = kspace_traj_complex/kspace_traj_complex.abs().max()/2 # normalize to -0.5,0.5
    kspace_traj_unique,  reverse_indices = np.unique( kspace_traj_complex.numpy(force=True), return_inverse=True, axis=0)
    kspace_traj_len = kspace_traj_unique.shape[0]

    print(reverse_indices)
    print(kspace_traj_unique[0:20])
    print(kspace_traj_complex[:10])
    r = kspace_traj_complex.abs().max()  # spoke radius in kspace, should be 0.5
    kspace_traj_augmented = torch.cat(
            (torch.from_numpy(kspace_traj_unique).to(device), r*1.005*torch.exp(1j*2*torch.pi*torch.arange(1, 257, device=device)/256)))
    # draw a circle around spokes
    # plt.scatter(kspace_traj_augmented.real,kspace_traj_augmented.imag,s=0.5)

    kspace_traj_augmented = torch.view_as_real(kspace_traj_augmented)
    vor = Voronoi(kspace_traj_augmented.numpy(force=True))
    regions_area = torch.zeros(len(vor.regions), device=device)
    for i, region in enumerate(vor.regions):
        if len(region) != 0:
            polygon = torch.from_numpy(vor.vertices[region])
            area = comp.polygon_area(polygon)
        else:
            area = float('inf')
        regions_area[i] = area
    regions_area = regions_area[vor.point_region][:kspace_traj_len]
    regions_area /= torch.sum(regions_area)
    regions_area = regions_area[reverse_indices]
    regions_area = eo.rearrange(
        regions_area, '(spokes_num spoke_len) -> spokes_num spoke_len', **spoke_shape)
    regions_area[:, spoke_shape['spoke_len']//2] /= spoke_shape['spokes_num']
    # Duplicate density for previously-removed points [i.e. DC points]
    return regions_area

    # fig = voronoi_plot_2d(vor,show_vertices=False,line_width=0.1,point_size=0.2)
    # plt.show()


def pipe_density_compensation(kspace_traj, im_size, grid_size):
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    w = tkbn.calc_density_compensation_function(
        ktraj=eo.rearrange(
            kspace_traj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)'),
        im_size=im_size,
        grid_size=grid_size)[0, 0]
    return eo.rearrange(w,
                        ' (spokes_num spoke_len) -> spokes_num spoke_len ', **spoke_shape)


def cihat_pipe_density_compensation(kspace_traj, im_size, grid_size, device=torch.device('cpu')):
    prev_device = kspace_traj.device
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    spoke_len = spoke_shape['spoke_len']
    omega = eo.rearrange(
        kspace_traj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)').to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(
        device)  # , grid_size=grid_size)
    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(device)
    # teop_ob = tkbn.ToepNufft()
    # teop_kernel = tkbn.calc_toeplitz_kernel(kspace_traj,im_size,norm = 'ortho')

    w = eo.repeat(
        torch.linspace(-1, 1-2/spoke_len, spoke_len, device=device).abs(),
        'spoke_len -> (spokes_num spoke_len)',
        **spoke_shape)
    tmp_  = torch.zeros((1,1,im_size[0],im_size[1]),dtype=torch.complex64,device = device)
    tmp_[0,0,im_size[0]//2,im_size[1]//2]=1
    # print(w.shape,omega.shape,im_size)
    w = w / pipe(
        tmp_,
        curry(nufft_ob.forward)(omega=omega,norm='ortho'),
        curry(torch.mul)(other=w),
        curry(adjnufft_ob.forward)(omega=omega,norm='ortho'),
        torch.abs,
        torch.max
    )
    return eo.rearrange(w,
                        '(spokes_num spoke_len) -> spokes_num spoke_len',
                        **spoke_shape).to(prev_device)

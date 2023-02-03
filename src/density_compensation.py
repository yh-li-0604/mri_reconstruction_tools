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



def voronoi_density_compensation(ktraj: torch.Tensor,
                                 im_size: Sequence[int],
                                 grid_size: Optional[Sequence[int]] = None):

    # spoke_shape = eo.parse_shape(ktraj, '_ spokes_num spoke_len')
    '''
    # ktraj_unique,  reverse_indices = np.unique(
    #     eo.rearrange(
    #             kspace_traj[0,0].numpy(), 'c spoke_num spoke_len -> (spoke_num spoke_len) c')
    #     ,  return_inverse=True, axis=0)
    # print(ktraj_unique.shape)'''
    ktraj_ = eo.rearrange(
        ktraj, 'c ktraj_len -> ktraj_len c').contiguous()
    device = ktraj_.device
    # plt.scatter(ktraj_unique[0],ktraj_unique[1],s=0.5)
    ktraj_complex = torch.view_as_complex(ktraj_)
    ktraj_len = ktraj_complex.shape[0]
    r = ktraj_complex.abs().max()  # spoke radius in kspace
    ktraj_augmented = torch.cat(
        (ktraj_complex, r*1.005*torch.exp(1j*2*torch.pi*torch.arange(1, 257, device=device)/256)))
    # draw a circle around spokes
    # plt.scatter(ktraj_augmented.real,ktraj_augmented.imag,s=0.5)
    ktraj_augmented = torch.view_as_real(ktraj_augmented)
    vor = Voronoi(ktraj_augmented.numpy(force=True))
    V, C = vor.vertices, vor.regions
    regions_area = torch.zeros(len(vor.regions), device=device)
    for i, region in enumerate(vor.regions):
        if len(region) != 0:
            polygon = torch.from_numpy(vor.vertices[region])
            area = comp.polygon_area(polygon)
        else:
            area = float('inf')
        regions_area[i] = area
    regions_area = regions_area[vor.point_region][:ktraj_len]
    regions_area /= torch.sum(regions_area)
    # Duplicate density for previously-removed points [i.e. DC points]
    # w = regions_area[reverse_indices]
    return regions_area

    # fig = voronoi_plot_2d(vor,show_vertices=False,line_width=0.1,point_size=0.2)
    # plt.show()


def pipe_density_compensation(ktraj, im_size, grid_size):
    spoke_shape = eo.parse_shape(ktraj, '_ spokes_num spoke_len')
    w = tkbn.calc_density_compensation_function(
        ktraj=eo.rearrange(
            ktraj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)'),
        im_size=im_size,
        grid_size=grid_size)[0, 0]
    return eo.rearrange(w,
                        ' (spokes_num spoke_len) -> spokes_num spoke_len ', **spoke_shape)


def cihat_pipe_density_compensation(ktraj, im_size, grid_size, device=torch.device('cpu')):
    prev_device = ktraj.device
    spoke_shape = eo.parse_shape(ktraj, '_ spokes_num spoke_len')
    spoke_len = spoke_shape['spoke_len']
    omega = eo.rearrange(
        ktraj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)').to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(
        device)  # , grid_size=grid_size)
    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(device)

    w = eo.repeat(
    torch.linspace(-1, 1-2/spoke_len, spoke_len, device=device).abs(),
        'spoke_len -> (spokes_num spoke_len)',
        **spoke_shape)

    # print(w.shape,omega.shape,im_size)    
    w = w / pipe(
        torch.ones((1,1)+im_size,dtype=torch.complex128,device=device),
        curry(nufft_ob.forward)(omega=omega),
        curry(torch.mul)(other=w),
        curry(adjnufft_ob.forward)(omega=omega),
        torch.abs,
        torch.max                
        )
    return eo.rearrange(w,
                        '(spokes_num spoke_len) -> spokes_num spoke_len', 
                        **spoke_shape).to(prev_device)

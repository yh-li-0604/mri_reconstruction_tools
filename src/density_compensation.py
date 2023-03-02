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
# import taichi as ti
from jax import vmap,pmap,jit
import jax.numpy as jnp
from jax.tree_util import tree_structure,tree_map
from src.torch_utils import jax_to_torch, torch_to_jax,as_complex,as_real

@jit
def augment(kspace_traj_unique):
    r = 0.5  # spoke radius in kspace, should be 0.5
    kspace_traj_augmented = jnp.concatenate(
            (kspace_traj_unique, r*1.005*jnp.exp(1j*2*jnp.pi*jnp.arange(1, 257)/256)))
    kspace_traj_augmented = as_real(kspace_traj_augmented)
    return kspace_traj_augmented

def voronoi_density_compensation(kspace_traj: torch.Tensor,
                                 im_size: Sequence[int],
                                 grid_size: Optional[Sequence[int]] = None,
                                 device = torch.device('cpu')):
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    kspace_traj_complex = eo.rearrange(
            kspace_traj, 'c spokes_num spoke_len -> (spokes_num spoke_len) c')
    kspace_traj_complex = torch.view_as_complex(kspace_traj_complex.contiguous()).to(device)
    kspace_traj_complex = torch_to_jax(kspace_traj_complex)
    kspace_traj_complex = kspace_traj_complex/jnp.abs(kspace_traj_complex).max()/2 # normalize to -0.5,0.5
    kspace_traj_unique,  reverse_indices = jnp.unique(kspace_traj_complex, return_inverse=True, axis=0)
    kspace_traj_len = kspace_traj_unique.shape[0]

    # draw a circle around spokes
    # plt.scatter(kspace_traj_augmented.real,kspace_traj_augmented.imag,s=0.5)
    kspace_traj_augmented = np.asarray(augment(kspace_traj_unique))
    vor = Voronoi(kspace_traj_augmented)
    def compute_area(region):
        if len(region) != 0:
            polygon = vor.vertices[region,]
            area = comp.polygon_area(polygon)
        else:
            area = float('inf')
        return area
    regions_area = jnp.array(tree_map(compute_area,vor.regions, is_leaf=lambda x:  len(x)==0 or isinstance(x[0], int)))
    regions_area = regions_area[vor.point_region][:kspace_traj_len]
    regions_area /= jnp.sum(regions_area)
    regions_area = regions_area[reverse_indices]
    regions_area = eo.rearrange(
        regions_area, '(spokes_num spoke_len) -> spokes_num spoke_len', **spoke_shape)
    regions_area = jax_to_torch(regions_area)
    regions_area[:, spoke_shape['spoke_len']//2] /= spoke_shape['spokes_num']
    # Duplicate density for previously-removed points [i.e. DC points]
    return regions_area

    # fig = voronoi_plot_2d(vor,show_vertices=False,line_width=0.1,point_size=0.2)
    # plt.show()


def pipe_density_compensation(kspace_traj, im_size,*args, **wargs):
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    w = tkbn.calc_density_compensation_function(
        ktraj=eo.rearrange(
            kspace_traj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)'),
        im_size=im_size)[0, 0]
    return eo.rearrange(w,
                        ' (spokes_num spoke_len) -> spokes_num spoke_len ', **spoke_shape)


def cihat_pipe_density_compensation(kspace_traj, nufft_ob, adjnufft_ob, device=torch.device('cpu'),*args, **wargs):
    im_size = adjnufft_ob.im_size.numpy(force=True)
    grid_size = adjnufft_ob.grid_size.numpy(force=True)
    prev_device = kspace_traj.device
    spoke_shape = eo.parse_shape(kspace_traj, '_ spokes_num spoke_len')
    spoke_len = spoke_shape['spoke_len']
    omega = eo.rearrange(
        kspace_traj, 'c spokes_num spoke_len -> c (spokes_num spoke_len)').to(device)

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


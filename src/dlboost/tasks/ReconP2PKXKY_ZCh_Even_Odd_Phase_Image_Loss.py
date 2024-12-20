
from httpx import patch
import napari
import zarr
from itertools import combinations_with_replacement, product, combinations
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
import einops as eo
# from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()

from typing import Optional, Sequence

import torchkbnufft as tkbn

from mrboost import computation as comp
from dlboost import losses
from dlboost.utils import complex_as_real_2ch, real_2ch_as_complex, complex_as_real_ch, to_png
from dlboost.tasks.boilerplate import *


class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        # self.nufft_op = tkbn.KbNufft(
        #     im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.patch_size = patch_size

    # @torch.compile()
    def forward(self, x):
        x_ = eo.rearrange(x, 'b ph d h w -> b d ph h w')
        x_ = self.recon_module(x_)
        return eo.rearrange(x_, 'b d ph h w -> b ph d h w')

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj_fixed, kspace_traj_moved = batch['kspace_traj'][:,
                                                                    0::2, ...], batch['kspace_traj'][:, 1::2, ...]
        kspace_data_compensated_fixed, kspace_data_compensated_moved = batch[
            'kspace_data_compensated'][:, 0::2, ...], batch['kspace_data_compensated'][:, 1::2, ...]

        image_init_fixed = nufft_adj_fn(
            kspace_data_compensated_fixed, kspace_traj_fixed, self.nufft_adj)
        image_init_moved = nufft_adj_fn(
            kspace_data_compensated_moved, kspace_traj_moved, self.nufft_adj)

        image_recon_fixed = self.forward(image_init_fixed)
        loss_f2m = self.calculate_recon_loss(image_recon=image_recon_fixed,
                                             image=image_init_moved)
        self.manual_backward(loss_f2m, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_f2m})

        image_recon_moved = self.forward(image_init_moved)
        loss_m2f = self.calculate_recon_loss(image_recon=image_recon_moved,
                                             image=image_init_fixed)
        self.manual_backward(loss_m2f, retain_graph=True)

        if self.global_step % 4 == 0:
            for i in range(image_init_moved.shape[1]):
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init_moved[0, i, 0, :, :])
                to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                       image_recon_moved[0, i, 0, :, :])
        recon_opt.step()

    def calculate_recon_loss(self, image_recon, image=None):
        loss = self.recon_loss_fn(torch.view_as_real(
            image_recon), torch.view_as_real(image))
        return loss

    def validation_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer

from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union
import napari
from numpy import diff, repeat
from torch.utils.hooks import RemovableHandle
import zarr
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
import einops as eo
import pdb
from matplotlib import pyplot as plt
import wandb

from monai.inferers import sliding_window_inference
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
        lambda_init=2,
        eta=1,
        weight_coef=1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.loss_recon_consensus_COEFF = 0.2
        self.lambda_init = lambda_init
        self.eta = eta
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.patch_size = patch_size
        self.weight_kspace_ema = 0
        # torch.zeros(640, requires_grad=False)
        # self.

    def forward(self, x):
        return self.recon_module(x)

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj = batch['kspace_traj']
        kspace_data = batch['kspace_data']
        kspace_data_compensated = batch['kspace_data_compensated']
        # print(kspace_traj.shape, kspace_data.shape, kspace_data_compensated.shape, w.shape)
        lambda_ = self.lambda_init + 0.00028125 * \
            self.global_step * kspace_data.shape[0]
        # generate a random discrete set for the kspace data
        masks = generate_disjoint_masks(kspace_data_compensated.shape[-2], [1]*15, kspace_data_compensated.device)
        reverse_masks = [torch.logical_not(m) for m in masks]

        kspace_data_compensated_masked_list = [
            eo.rearrange(kspace_data_compensated[..., mask, :], "b ph z sp len -> b ph z (sp len)") for mask in reverse_masks]
        # kspace_data_masked_list = [kspace_data[..., mask] for mask in masks]
        kspace_traj_masked_list = [eo.rearrange(kspace_traj[..., mask, :], "b ph c sp len -> b ph c (sp len)")
                                   for mask in reverse_masks]
        # w_masked_list = [w[...,mask] for mask in masks]

        kspace_data_estimated_blind = torch.zeros_like(kspace_data)
        image_init, image_recon = None, None
        for k_compensated, k_traj, mask in zip(kspace_data_compensated_masked_list, kspace_traj_masked_list, masks):
            # print(k_compensated.shape, k_traj.shape)
            image_init = nufft_adj_fn(k_compensated, k_traj, self.nufft_adj)
            image_recon = self.forward(image_init)
            kspace_data_estimated = nufft_fn(image_recon,
                                                  eo.rearrange(kspace_traj[..., mask, :], "b ph z sp len -> b ph z (sp len)"),
                                                  self.nufft_op)
            kspace_data_estimated_blind[..., mask, :] = eo.rearrange(
                kspace_data_estimated, "b ph z (sp len) -> b ph z sp len", sp=torch.sum(mask))
        with torch.no_grad():
            image_init_unblind = nufft_adj_fn(
                eo.rearrange(kspace_data_compensated,
                             "b ph z sp len -> b ph z (sp len)"),
                eo.rearrange(kspace_traj, "b ph z sp len -> b ph z (sp len)"),
                self.nufft_adj)
            image_recon_unblind = self.forward(image_init_unblind)
            kspace_data_estimated_unblind = nufft_fn(
                image_recon_unblind,
                eo.rearrange(kspace_traj, "b ph c sp len -> b ph c (sp len)"),
                self.nufft_op)
            kspace_data_estimated_unblind = eo.rearrange(
                kspace_data_estimated_unblind, "b ph z (sp len) -> b ph z sp len", sp=kspace_data.shape[-2])

        # weight = batch["kspace_density_compensation"] * 1e4
        # weight = weight**self.weight_coef
        # eta = 0.1
        weight = torch.arange(1, kspace_data.shape[-1]//2+1, device=kspace_data.device)
        # weight = torch.cat([weight, ], dim=0)
        weight_reverse_sample_density = torch.cat([weight.flip(0),weight], dim=0)
        # weight_reverse_sample_density = 1.0
        # weight_reverse_sample_density  = torch.log(weight_reverse_sample_density)+1
        # weight_reverse_sample_density  = torch.log(weight_reverse_sample_density)+1

        diff_revisit = (kspace_data_estimated_blind + \
            lambda_ * kspace_data_estimated_unblind - \
            (lambda_+1) * kspace_data)*weight_reverse_sample_density
        # self.weight_kspace_ema = eta * torch.mean(diff_revisit.clone().detach(), (0,1,2,3)) + (1-eta)*self.weight_kspace_ema
        # print(self.weight_kspace_ema.abs())
        # print(f.interpolate(self.weight_kspace_ema.abs(),1/16))
        # self.weight_kspace_ema = 1+1j
        # diff_revisit *= (weight_reverse_sample_density*self.weight_kspace_ema.abs())
        # loss_revisit = diff_revisit*diff_revisit.conj() #L2 distance between 2 kspace point is the square of the distance between their real and imaginary part
        loss_revisit = torch.mean(diff_revisit*diff_revisit.conj())

        # loss_reg = self.eta * (torch.view_as_real(kspace_data_estimated_blind*weight)
        #                                 -torch.view_as_real(kspace_data_estimated_unblind*weight))**2
        
        diff_reg = (kspace_data_estimated_blind - kspace_data_estimated_unblind)*weight_reverse_sample_density
        # diff_reg *= (weight_reverse_sample_density*self.weight_kspace_ema.abs())
        # loss_reg = self.eta * diff_reg*diff_reg.conj()
        loss_reg = torch.mean(self.eta * diff_reg * diff_reg.conj())

        self.manual_backward(loss_revisit+loss_reg, retain_graph=True)
        self.log_dict({"recon/loss_revisit": loss_revisit})
        self.log_dict({"recon/loss_reg": loss_reg})
        self.log_dict({"recon/recon_loss": loss_reg+loss_revisit})
        if self.global_step % 4 == 0:
            for i in range(image_init.shape[1]):
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init[0, i, 0, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_blind_ph{i}.png',
                       image_recon[0, i, 0, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_unblind_ph{i}.png',
                       image_recon_unblind[0, i, 0, :, :])  # , vmin=0, vmax=2)
        recon_opt.step()

    def validation_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer
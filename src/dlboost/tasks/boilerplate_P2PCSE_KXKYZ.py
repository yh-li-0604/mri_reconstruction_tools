from typing import Any, Sequence

import lightning as L
import torch
import torchkbnufft as tkbn
import zarr
from dlboost.utils import formap, to_png
from dlboost.utils.tensor_utils import interpolate
from einops import rearrange  # , reduce, repeat
from monai.inferers import PatchInferer, SlidingWindowSplitter
from torch import nn, optim
from torch.nn import functional as f
from dlboost.models import ComplexUnet, DWUNet

# def cse_forward(image_init_ch, downsample, upsample, ch_pad, cse_module):
#     ph,ch = image_init_ch.shape[0:2]
#     image_init_ch_lr = image_init_ch.clone()
#     for i in range(3):
#         image_init_ch_lr = downsample(image_init_ch_lr)
#     # image_init_ch_lr = interpolate(image_init_ch, scale_factor=0.25, mode='bilinear')
#     if ch < ch_pad:
#         image_init_ch_lr = f.pad(image_init_ch_lr, (0,0,0,0,0,ch_pad-ch))
#     # print(image_init_ch_lr.shape)
#     csm_lr = cse_module(
#         rearrange(image_init_ch_lr, 'ph ch h w -> () (ph ch) h w'))
#     csm_lr = rearrange(csm_lr, '() (ph ch) h w -> ph ch h w', ph=ph)[:, :ch]
#     csm_hr = csm_lr
#     for i in range(3):
#         csm_hr = upsample(csm_hr)
#     # csm_hr = self.upsample(csm_lr)
#     # csm_hr = interpolate(csm_lr, scale_factor=4, mode='bilinear')
#     # devide csm by its root square of sum
#     csm_hr_norm = csm_hr / \
#         torch.sqrt(torch.sum(torch.abs(csm_hr)**2, dim=1, keepdim=True))
#     return csm_hr_norm


def postprocessing(x):
    x[:, :, [0, 3], ...] = 0
    return 2 * x  # compensate for avg merger from monai


def nufft_adj_gpu(
    kspace_data, csm, kspace_traj, nufft_adj, inference_device, storage_device
):
    image_init = nufft_adj(
        kspace_data.to(inference_device), kspace_traj.to(inference_device)
    )
    result_reduced = torch.sum(image_init * csm.conj().to(inference_device), dim=1)
    return result_reduced.to(storage_device)


def forward_contrast(
    kspace_data,
    kspace_traj,
    kspace_data_cse,
    kspace_traj_cse,
    recon_module,
    cse_forward,
    nufft_adj,
    inference_device,
    storage_device,
    inferer=None,
):
    """
    kspace_data: [ph, ch, z, len]
    kspace_traj: [ph, 2, len]
    """
    with torch.no_grad():
        csm = nufft_adj(
            kspace_data_cse.to(inference_device), kspace_traj_cse.to(inference_device)
        )
        # input of cse_forward shape is [ph, ch, d, h, w]
        csm = cse_forward(csm).to(storage_device)
        image_init = formap(nufft_adj_gpu, 2, 1, 10)(
            kspace_data,
            csm,
            kspace_traj=kspace_traj,
            nufft_adj=nufft_adj,
            inference_device=inference_device,
            storage_device=storage_device,
        )
        if inferer is None:
            image_recon = recon_module(image_init.unsqueeze(0)).squeeze(0)
        else:
            image_recon = inferer(image_init.unsqueeze(0), recon_module).squeeze(0)
    return image_recon, image_init, csm


def gradient_loss(s, penalty="l2", reduction="mean"):
    if s.ndim != 4:
        raise RuntimeError(f"Expected input `s` to be an 4D tensor, but got {s.shape}")
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    if penalty == "l2":
        dy = dy * dy
        dx = dx * dx
    elif penalty == "l1":
        pass
    else:
        raise NotImplementedError
    if reduction == "mean":
        d = torch.mean(dx) + torch.mean(dy)
    elif reduction == "sum":
        d = torch.sum(dx) + torch.sum(dy)
    return d / 2.0


class P2PCSE_KXKYZ(L.LightningModule):
    def __init__(
        self,
        nufft_im_size=(320, 320),
        patch_size=(5, 64, 64),
        ch_pad=42,
        lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["recon_module", "cse_module", "regis_module", "recon_loss_fn"]
        )
        self.recon_module = ComplexUnet(
            in_channels=5,
            out_channels=5,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=10,
                out_channels=10,
                spatial_dims=3,
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=((3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7)),
            ),
        )
        self.cse_module = ComplexUnet(
            in_channels=ch_pad,
            out_channels=ch_pad,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2 * ch_pad,
                out_channels=2 * ch_pad,
                spatial_dims=3,
                strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1)),
                kernel_sizes=((3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7)),
                features=(128, 256, 256, 256, 256),
            ),
        )
        self.automatic_optimization = False
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = torch.nn.L1Loss(reduction="none")
        self.recon_lr = lr
        self.recon_optimizer = optim.AdamW
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)
        self.teop_op = tkbn.ToepNufft()
        self.patch_size = patch_size
        self.ch_pad = ch_pad
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )
        self.inferer = PatchInferer(
            SlidingWindowSplitter(patch_size, overlap=(0.5, 0, 0), offset=(1, 0, 0)),
            batch_size=8,
            preprocessing=lambda x: x.to(self.device),
            postprocessing=lambda x: postprocessing(x).to(torch.device("cpu")),
            value_dtype=torch.complex64,
        )

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        batch = batch[0]
        # kspace data is in the shape of [ch, ph, sp]
        kspace_traj_fixed, kspace_traj_moved = (
            batch["kspace_traj_fixed"],
            batch["kspace_traj_moved"],
        )
        kspace_traj_cse_fixed, kspace_traj_cse_moved = (
            batch["kspace_traj_cse_fixed"],
            batch["kspace_traj_cse_moved"],
        )
        kspace_data_fixed, kspace_data_moved = (
            batch["kspace_data_z_fixed"],
            batch["kspace_data_z_moved"],
        )
        kspace_data_compensated_fixed, kspace_data_compensated_moved = (
            batch["kspace_data_z_compensated_fixed"],
            batch["kspace_data_z_compensated_moved"],
        )
        kspace_data_cse_fixed, kspace_data_cse_moved = (
            batch["kspace_data_z_cse_fixed"],
            batch["kspace_data_z_cse_moved"],
        )

        # # kspace weighted loss
        weight = torch.arange(
            1, kspace_data_fixed.shape[-1] // 2 + 1, device=kspace_data_fixed.device
        )
        weight_reverse_sample_density = torch.cat([weight.flip(0), weight], dim=0)

        image_init_fixed_ch = self.nufft_adj_forward(
            kspace_data_cse_fixed, kspace_traj_cse_fixed
        )
        # image_init_fixed_ch = self.nufft_adj_forward(
        #     kspace_data_fixed, kspace_traj_fixed)
        # shape is [1, ch, d, h, w]

        csm_fixed = self.cse_forward(image_init_fixed_ch).expand(5, -1, -1, -1, -1)
        # csm_fixed = self.cse_forward(image_init_fixed_ch)
        # csm_smooth_loss = self.smooth_loss_coef * self.smooth_loss_fn(csm_fixed)
        # self.log_dict({"recon/csm_smooth_loss": csm_smooth_loss})

        image_init_fixed = self.nufft_adj_forward(
            kspace_data_compensated_fixed, kspace_traj_fixed
        )
        image_init_fixed = torch.sum(image_init_fixed * csm_fixed.conj(), dim=1)
        # shape is [ph, d, h, w]
        image_recon_fixed = self.recon_module(image_init_fixed.unsqueeze(0)).squeeze(0)
        loss_f2m = self.calculate_recon_loss(
            image_recon=image_recon_fixed.unsqueeze(1).expand_as(csm_fixed),
            csm=csm_fixed,
            kspace_traj=kspace_traj_moved,
            kspace_data=kspace_data_moved,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_f2m, retain_graph=True)
        # self.manual_backward(loss_f2m+csm_smooth_loss, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_f2m})

        image_init_moved_ch = self.nufft_adj_forward(
            kspace_data_cse_moved, kspace_traj_cse_moved
        )
        # if self.global_step % 4 == 0:
        #     for ch in [0, 3, 5]:
        #         to_png(self.trainer.default_root_dir+f'/image_init_moved_ch{ch}.png',
        #                image_init_moved_ch[0, ch, 0, :, :])  # , vmin=0, vmax=2)
        # image_init_moved_ch = self.nufft_adj_forward(
        #     kspace_data_moved, kspace_traj_moved)
        # shape is [ph, ch, h, w]
        csm_moved = self.cse_forward(image_init_moved_ch).expand(5, -1, -1, -1, -1)
        # csm_moved = self.cse_forward(image_init_moved_ch)
        # csm_smooth_loss = self.smooth_loss_coef * self.smooth_loss_fn(csm_moved)

        image_init_moved = self.nufft_adj_forward(
            kspace_data_compensated_moved, kspace_traj_moved
        )
        image_init_moved = torch.sum(image_init_moved * csm_moved.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_moved = self.recon_module(image_init_moved.unsqueeze(0)).squeeze(
            0
        )  # shape is [ph, h, w]
        loss_m2f = self.calculate_recon_loss(
            image_recon=image_recon_moved.unsqueeze(1).expand_as(csm_moved),
            csm=csm_moved,
            kspace_traj=kspace_traj_fixed,
            kspace_data=kspace_data_fixed,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_m2f, retain_graph=True)
        # self.manual_backward(loss_m2f + csm_smooth_loss, retain_graph=True)

        if self.global_step % 4 == 0:
            for ch in [0, 3, 5]:
                to_png(
                    self.trainer.default_root_dir + f"/image_init_moved_ch{ch}.png",
                    image_init_moved_ch[0, ch, 0, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/csm_moved_ch{ch}.png",
                    csm_moved[0, ch, 0, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init_moved.shape[0]):
                to_png(
                    self.trainer.default_root_dir + f"/image_init_ph{i}.png",
                    image_init_moved[i, 0, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/image_recon_ph{i}.png",
                    image_recon_moved[i, 0, :, :],
                )  # , vmin=0, vmax=2)
        recon_opt.step()

    def cse_forward(self, image_init_ch):
        ph, ch = image_init_ch.shape[0:2]
        image_init_ch_lr = image_init_ch.clone()
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        # image_init_ch_lr = interpolate(image_init_ch, scale_factor=0.25, mode='bilinear')
        if ch < self.ch_pad:
            image_init_ch_lr = f.pad(
                image_init_ch_lr, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        # print(image_init_ch_lr.shape)
        csm_lr = self.cse_module(image_init_ch_lr)
        csm_hr = csm_lr[:, :ch]
        for i in range(3):
            csm_hr = self.upsample(csm_hr)
        # csm_hr = self.upsample(csm_lr)
        # csm_hr = interpolate(csm_lr, scale_factor=4, mode='bilinear')
        # devide csm by its root square of sum
        csm_hr_norm = csm_hr / torch.sqrt(
            torch.sum(torch.abs(csm_hr) ** 2, dim=1, keepdim=True)
        )
        return csm_hr_norm

    def nufft_forward(self, image_init, kspace_traj):
        ph, ch, d, h, w = image_init.shape
        kspace_data = self.nufft_op(
            rearrange(image_init, "ph ch d h w -> ph (ch d) h w"),
            kspace_traj,
            norm="ortho",
        )
        return rearrange(kspace_data, "ph (ch d) len -> ph ch d len", ch=ch)

    def nufft_adj_forward(self, kspace_data, kspace_traj):
        ph, ch, d, length = kspace_data.shape
        # breakpoint()
        image = self.nufft_adj(
            rearrange(kspace_data, "ph ch d len -> ph (ch d) len"),
            kspace_traj,
            norm="ortho",
        )
        return rearrange(image, "ph (ch d) h w -> ph ch d h w", ch=ch)

    def calculate_recon_loss(
        self, image_recon, csm, kspace_traj, kspace_data, weight=None
    ):
        # kernel = tkbn.calc_toeplitz_kernel(kspace_traj, im_size=self.nufft_im_size)
        # image_HTH = self.teop_op(image_recon, kernel, smaps = csm, norm= 'ortho')
        # image_HT = self.nufft_adj(kspace_data, kspace_traj, smaps = csm, norm= 'ortho')
        kspace_data_estimated = self.nufft_forward(image_recon * csm, kspace_traj)

        loss_not_reduced = self.recon_loss_fn(
            torch.view_as_real(weight * kspace_data_estimated),
            torch.view_as_real(kspace_data * weight),
        )
        # self.recon_loss_fn(torch.view_as_real(
        #     image_HTH), torch.view_as_real(image_HT))
        loss = torch.mean(loss_not_reduced)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        for b in batch:
            image_recon, image_init, csm = forward_contrast(
                b["kspace_data_z_compensated"],
                b["kspace_traj"],
                b["kspace_data_z_cse"],
                b["kspace_traj_cse"],
                # b["kspace_data_z"], b["kspace_traj"],
                recon_module=self.recon_module,
                cse_forward=self.cse_forward,
                nufft_adj=self.nufft_adj_forward,
                inference_device=self.device,
                storage_device=torch.device("cpu"),
                inferer=self.inferer,
            )
            print(image_recon.shape, image_init.shape, csm.shape)
            zarr.save(
                self.trainer.default_root_dir
                +
                #   f'/epoch_{self.trainer.current_epoch}'+f'/image_init.zarr', image_init.abs().numpy(force=True))
                f"/epoch_{self.trainer.current_epoch}"
                + "/image_init.zarr",
                image_init[:, 35:45].abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                +
                #   f'/epoch_{self.trainer.current_epoch}'+f'/image_recon.zarr', image_recon.abs().numpy(force=True))
                f"/epoch_{self.trainer.current_epoch}"
                + "/image_recon.zarr",
                image_recon[:, 35:45].abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                +
                #   f'/epoch_{self.trainer.current_epoch}'+f'/csm.zarr', csm.abs().numpy(force=True))
                f"/epoch_{self.trainer.current_epoch}"
                + "/csm.zarr",
                csm[:, :, 35:45].abs().numpy(force=True),
            )
            print(
                "Save image_init, image_recon, csm to "
                + self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
            )
            for ch in [0, 3, 5]:
                # to_png(self.trainer.default_root_dir+f'/epoch_{self.trainer.current_epoch}_'+f'/image_init_moved_ch{ch}.png',
                #        image_init_ch[0, ch, :, :])  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/csm_moved_ch{ch}.png",
                    csm[0, ch, 40, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init.shape[0]):
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_init_ph{i}.png",
                    image_init[i, 40, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_recon_ph{i}.png",
                    image_recon[i, 40, :, :],
                )  # , vmin=0, vmax=2)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        device=torch.device("cuda"),
        ch_reduce_fn=torch.sum,
    ) -> Any:
        results = []
        for b in batch:
            # print(b["kspace_data_z"].shape, b["kspace_traj"].shape)
            image_recon, image_init, csm = forward_contrast(
                b["kspace_data_z_compensated"],
                b["kspace_traj"],
                b["kspace_data_z_cse"],
                b["kspace_traj_cse"],
                recon_module=self.recon_module,
                cse_forward=self.cse_forward,
                nufft_adj=self.nufft_adj_forward,
                inference_device=self.device,
                storage_device=torch.device("cpu"),
                inferer=self.inferer,
            )
            # print(image_recon.shape, image_init.shape, csm.shape)
            results.append(
                {
                    "image_recon": image_recon,
                    "image_init": image_init,
                    "csm": csm,
                    "id": b["id"],
                }
            )
        return results

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            [
                {"params": self.recon_module.parameters()},
                {"params": self.cse_module.parameters()},
            ],
            lr=self.recon_lr,
        )
        return recon_optimizer

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    # if self.trainer.training:
    #     batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    # else:
    #     batch = batch
    # return batch

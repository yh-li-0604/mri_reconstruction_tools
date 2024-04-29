import numpy as np
import torch
import torchkbnufft as tkbn

from mrboost.computation import generate_nufft_op, nufft_2d, nufft_adj_2d


def test_nufft2d_adj():
    data = torch.view_as_complex(
        torch.stack((torch.ones(10, 1, 12), torch.ones(10, 1, 12)), dim=-1)
    )
    omega = torch.rand(2, 12) * 2 * np.pi - np.pi

    adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
    image_tkbn = adjkb_ob(data.clone(), omega.clone(), norm="ortho")

    image_finufft = nufft_adj_2d(data, omega, (8, 8))
    assert torch.allclose(image_tkbn, image_finufft, 1e-2, 1e-4)


def test_nufft_2d():
    image = torch.view_as_complex(
        torch.stack((torch.ones(10, 15, 8, 8), torch.ones(10, 15, 8, 8)), dim=-1)
    )
    omega = torch.rand(2, 24) * 2 * np.pi - np.pi
    kb_ob = tkbn.KbNufft(im_size=(8, 8))
    data_tkbn = kb_ob(image.clone(), omega.clone(), norm="ortho")

    data_finufft = nufft_2d(image, omega, (8, 8))
    assert torch.allclose(data_tkbn, data_finufft, 1e-2, 1e-4)


def data_binning_test():
    data = torch.rand((15, 80, 2550, 640))
    spokes_per_contra = 75
    phase_num = 5
    spokes_per_phase = 15
    contrast_num = 34
    sorted_r_idx = torch.stack(
        [torch.randperm(spokes_per_contra) for i in range(contrast_num)]
    )
    # o = data_binning_jl(
    #     data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase
    # )
    # return o.shape


# jl.data_binning(jl.Array(data.numpy()), jl.Array(sorted_r_idx.numpy()), contrast_num, spokes_per_contra, phase_num, spokes_per_phase)
# print(data_binning_test())
if __name__ == "__main__":
    test_nufft2d_adj()
    test_nufft_2d()
    print("All tests passed!")

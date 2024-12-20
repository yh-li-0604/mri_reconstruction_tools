from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    NotRequired,
    Sequence,
    Tuple,
    TypedDict,
)

import torch
import torch.nn.functional as f
from jaxtyping import PyTree
from optree import tree_map
from plum import dispatch, overload
from torch import Size, Tensor
from xarray import DataArray

Location = Sequence[slice]
Data_With_Location = Tuple[Any, Location]
PadSizes = Sequence[Tuple[int, int]]
NamedSize = Dict[str, int]
Patches_With_PadSizes = TypedDict(
    "Patches_With_PadSizes",
    {
        "patches": Sequence[Data_With_Location],
        "pad_sizes": NotRequired[PadSizes],
        "padded_size": NotRequired[Size],
    },
)

XArrayDevice = Literal["xarray", "disk", "xa"]


@overload
def _transfer_to_device(data: Tensor, device: torch.device) -> Tensor:
    return data.to(device)


@overload
def _transfer_to_device(
    data: Data_With_Location, device: torch.device
) -> Data_With_Location:
    return (_transfer_to_device(data[0], device), data[1])


@overload
def _transfer_to_device(
    data: Dict[str, Data_With_Location | Tensor], device: torch.device
) -> Dict[str, Data_With_Location | Tensor]:
    return {k: _transfer_to_device(v, device) for k, v in data.items()}


@overload
def _transfer_to_device(
    data: PyTree[DataArray, "T"], device: torch.device
) -> PyTree[Tensor, "T"]:
    return tree_map(lambda x: torch.from_numpy(x.values).to(device), data)


# @overload
# def _transfer_to_device(
#     data: Tensor,
#     device: XArrayDevice,
#     dims: Sequence[str],
# ) -> PyTree[DataArray, "T"]:
#     return DataArray( data= data.cpu().numpy(),dims = dims)


@overload
def _transfer_to_device(
    data: PyTree[Tensor, "T"],
    device: XArrayDevice,
    dims: PyTree[Sequence[str], "T"],
) -> PyTree[DataArray, "T"]:
    return tree_map(lambda x, d: DataArray(data=x.cpu().numpy(), dims=d), data, dims)


@dispatch
def _transfer_to_device(data, device):
    pass




def pad(data: Tensor, dims, pad_sizes, mode="constant", value=0):
    # Create a padding configuration tuple
    # The tuple should have an even number of elements, with the form:
    # (pad_left, pad_right, pad_top, pad_bottom, ...)
    # Since we want to pad a specific dimension 'dim', we need to calculate
    # the correct positions in the tuple for pad_size_1 and pad_size_2.

    # Initialize padding configuration with zeros for all dimensions
    pad_config = [0] * (data.dim() * 2)
    # Set the padding sizes for the specified dimensions
    for dim, (pad_size_1, pad_size_2) in zip(dims, pad_sizes):
        # Calculate the indices in the padding configuration tuple
        # PyTorch pads from the last dimension backwards, so we need to invert the dimension index
        pad_index_1 = -(dim + 1) * 2
        pad_index_2 = pad_index_1 + 1

        # Set the desired padding sizes at the correct indices
        pad_config[pad_index_1] = pad_size_1
        pad_config[pad_index_2] = pad_size_2

    # Convert the list to a tuple and apply padding
    pad_config = tuple(pad_config)
    return f.pad(data, pad_config, mode=mode, value=value)


def pad(data: DataArray, pad_sizes, mode="constant", value=0):
    # Apply padding using xarray's pad function
    # common_keys = data.dims & pad_sizes.keys()
    # _pad_sizes = {k: pad_sizes[k] for k in common_keys}
    padded_DataArray = data.pad(pad_sizes, mode=mode, constant_values=value)
    return padded_DataArray


@overload
def crop(data: Tensor, dims, start_indices, crop_sizes) -> Tensor:
    # Initialize slicing configuration with colons for all dimensions
    slice_config = [slice(None)] * data.dim()

    # Set the slicing configuration for the specified dimensions
    for dim, start_index, crop_size in zip(dims, start_indices, crop_sizes):
        # crop_size = crop_sizes[dim]
        slice_config[dim] = slice(start_index, start_index + crop_size)

    # Apply slicing
    return data[tuple(slice_config)]


# @overload
# def crop(
#     data: DataArray, start_indices: Dict[str, int], crop_sizes: Dict[str, int]
# ) -> DataArray:
#     # Apply cropping using xarray's isel function
#     slices = {k: slice(start_indices[k], start_indices[k] + crop_sizes[k]) for k in start_indices.keys()}
#     return data.isel(slices)


@overload
def crop(
    data: PyTree[DataArray, "T"],
    start_indices: PyTree[Dict[str, int], "T"],
    crop_sizes: PyTree[Dict[str, int], "T"],
) -> PyTree[DataArray, "T"]:
    slices = tree_map(lambda x, y: slice(x, x + y), start_indices, crop_sizes)
    return tree_map(lambda x, y: x.isel(y), data, slices)


@dispatch
def crop(data, dims, start_indices, crop_sizes):
    pass

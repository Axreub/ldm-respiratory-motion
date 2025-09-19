import torch
import torch.nn.functional as F
from typing import List, Tuple


def load_activation(activation: str = "elu") -> torch.nn.Module:
    """Function that loads in activation functions. Supported values are 'elu', 'gelu' and 'tanh'."""
    if activation.lower() == "elu":
        activation = torch.nn.ELU()
    elif activation.lower() == "gelu":
        activation = torch.nn.GELU()
    elif activation.lower() == "tanh":
        activation = torch.nn.Tanh()
    elif activation.lower() == "identity":
        activation = torch.nn.Identity()
    else:
        raise ValueError(
            f"Invalid activation function: {activation}. Supported values are 'elu' and 'gelu' and 'tanh'."
        )

    return activation


def load_norm(channels, norm: str | None = None) -> torch.nn.Module:
    """Function that loads in normalization layers. Supported values are 'batch', 'instance' or 'group'. Defaults to None."""
    if norm.lower() == "batch":
        norm_layer = torch.nn.BatchNorm3d(num_features=channels)
    elif norm.lower() == "instance":
        norm_layer = torch.nn.InstanceNorm3d(num_features=channels, affine=True)
    elif norm.lower() == "group":
        norm_layer = torch.nn.GroupNorm(
            num_groups=max(1, channels // 8), num_channels=channels, affine=True
        )
    elif norm == "identity":
        norm_layer = torch.nn.Identity()
    else:
        raise ValueError(
            f"Invalid normalization layer: {norm}. Supported values are 'batch', 'group', 'identity' and 'instance'."
        )

    return norm_layer


def create_pads(keep_z: List[bool], img_width=256, img_depth=50) -> Tuple[List[int]]:
    """Creates padding lists for each layer based on keep_z and channel_multipliers."""
    pads_xy, pads_z = [], []
    z_range = 0
    for i, keep in enumerate(keep_z):
        xy = img_width
        xy = [xy := xy // 2 if idx != i else xy % 2 for idx in range(i + 1)][-1]
        pads_xy.append(xy)
        if keep:
            pads_z.append(0)
        else:
            z_range += 1
            z = img_depth
            z = [
                z := z // 2 if idx != z_range - 1 else z % 2 for idx in range(z_range)
            ][-1]
            pads_z.append(z)

    return pads_xy, pads_z

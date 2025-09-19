import torch
import torch.nn.functional as F
import airlab as al
import numpy as np
from typing import Tuple, Union

from utils.ct_views import view_to_axis
from utils.tensor_utils import unsqueeze_channels


def torch_to_airlab(
    t: torch.Tensor,
    spacing: Tuple[float, float, float],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "al.Image":
    """
    Converts a PyTorch tensor to an airlab Image object.

    Args:
        t (torch.Tensor): The input tensor. Can be 3D (D, H, W) or 4D (C, D, H, W).
        spacing (Tuple[float, float, float]): The voxel spacing for the image in (x,y,z) order.
        device (Union[str, torch.device], optional): The device to move the tensor to. Defaults to "cpu".
        dtype (torch.dtype, optional): The data type for the tensor. Defaults to torch.float32.
        origin (Tuple[float, float, float], optional): The origin of the image. Defaults to (0.0, 0.0, 0.0).

    Returns:
        airlab.Image: The airlab Image object created from the tensor.
    """

    t = t.to(device, dtype, non_blocking=True)
    if t.ndim == 3:
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    elif t.ndim == 4:
        t = t.unsqueeze(0)  # (1,C,D,H,W)
    size = t.shape[-3:]
    return al.Image(t, size, spacing, origin)


def get_dvf_from_bspline(
    bspline: "al.transformation.pairwise.BsplineTransformation",
) -> torch.Tensor:
    """Get a displacement field from an airlab B-spline transform."""
    dvf = bspline.get_displacement()
    dvf = dvf.permute(3, 0, 1, 2).contiguous()  # (D,H,W,C) -> (C,D,H,W)
    return dvf


def save_bspline(
    bspline_transform: "al.transformation.pairwise.BsplineTransformation",
    path: str,
    order: int,
) -> None:
    """Save an airlab B-spline transform to a torch file."""
    torch.save(
        {
            "state_dict": bspline_transform.state_dict(),
            "image_size": bspline_transform._image_size,
            "sigma": bspline_transform._stride,
            "order": order,
            "diffeomorphic": bspline_transform._diffeomorphic,
        },
        path,
    )


def load_bspline(path: str) -> "al.transformation.pairwise.BsplineTransformation":
    """Load an airlab B-spline transform from a torch file."""
    data = torch.load(path, weights_only=False)
    bspline_transform = al.transformation.pairwise.BsplineTransformation(
        image_size=data["image_size"],
        sigma=data["sigma"],
        order=data["order"],
        diffeomorphic=data["diffeomorphic"],
        device="cpu",
    )
    bspline_transform.eval()
    bspline_transform.load_state_dict(data["state_dict"])
    for p in bspline_transform.parameters():
        p.requires_grad = False

    return bspline_transform


def get_dvf_components(
    dvf: np.ndarray, view: str, idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the plane corresponding to the requested axis view (axial, coronal, sagittal).
    Used for plotting DVF components for the requested axis view.

    Args:
        dvf (np.ndarray): The displacement field. First channel is z, second is y, third is x.
        view (str): The view to return the components of.
        idx (int): The index of the slice to return the components of.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The row and column components of the displacement field.
    """

    axis = view_to_axis(view)
    if axis == 0:  # axial
        displacement_rows = dvf[1, idx]
        displacement_cols = dvf[2, idx]

    elif axis == 1:  # coronal
        displacement_rows = dvf[0, :, idx]
        displacement_cols = dvf[2, :, idx]

    elif axis == 2:  # sagittal
        displacement_rows = dvf[0, :, :, idx]
        displacement_cols = dvf[1, :, :, idx]

    else:
        raise ValueError
    return displacement_rows, displacement_cols


def warp_image(
    image: torch.Tensor,
    dvf: torch.Tensor,
    spacing: Tuple[float, float, float],
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """
    Warps an image using a displacement field.

    Args:
        image (torch.Tensor): The image to warp. Shape: [D,H,W]
        dvf (torch.Tensor): The displacement field to warp the image with. Shape: [3,D,H,W]
        spacing (Tuple[float, float, float]): The spacing of the image in (z,y,x) order.
        device (Union[torch.device, str], optional): The device to move the tensors to. Defaults to "cpu".

    Returns:
        torch.Tensor: The warped image. Shape: [D,H,W]
    """

    # airlab expects spacing in (x,y,z) order
    spacing = spacing[::-1]

    if image.ndim == 3:
        image = unsqueeze_channels(image, device=image.device)
    image = torch_to_airlab(image, spacing=spacing, device=device)

    dvf = dvf.permute(1, 2, 3, 0).contiguous()  # (C,D,H,W) -> (D,H,W,C)

    grid = al.transformation.utils.compute_grid(
        image.size, dtype=image.dtype, device=image.device
    )

    warped_image = F.grid_sample(
        image.image, dvf + grid, padding_mode="border", mode="bilinear"
    )

    return warped_image

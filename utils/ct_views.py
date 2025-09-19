import numpy as np
import torch


def get_ct_view(
    vol: np.ndarray | torch.Tensor, view: str, idx: int
) -> np.ndarray | torch.Tensor:
    """
    Return a single CT slice along the requested anatomical view.

    The last three dims must always be (D,H,W).

    Args:
        vol   : np.ndarray | torch.Tensor
        view  : "axial"|"coronal"|"sagittal"
        idx   : slice index in that view

    Returns:
        The 2-D slice with all leading dims preserved.
        Shape examples: (H,W), (C,H,W) or (B,C,H,W)
    """
    axis = view_to_axis(view)

    # Convert to an absolute dim (axial => dim = -3, coronal => dim = -2, sagittal => dim = -1)
    dim = vol.ndim - 3 + axis

    if isinstance(vol, torch.Tensor):
        return torch.select(vol, dim, idx)
    else:
        slc = [slice(None)] * vol.ndim
        slc[dim] = idx
        return vol[tuple(slc)]


def view_to_axis(view: str) -> int:
    """
    Returns the axis corresponding to the requested view (axial, coronal, sagittal).
    """
    if view == "axial":
        return 0
    if view == "coronal":
        return 1
    if view == "sagittal":
        return 2
    raise ValueError(
        f"Invalid view: {view}. Accepted views are 'axial', 'coronal', and 'sagittal'."
    )

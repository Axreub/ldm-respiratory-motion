import numpy as np
import torch
import torch.nn.functional as F


def normalize_ct_image(
    image: np.ndarray, min_hu: int = -1024, max_hu: int = 3071
) -> np.ndarray:
    """Normalize CT image to [-1, 1] range."""
    image = np.clip(image, min_hu, max_hu)
    return 2 * (image - min_hu) / (max_hu - min_hu) - 1


def resample_volume(
    vol: torch.Tensor, target_dimensions: tuple[int, int, int]
) -> torch.Tensor:
    """
    Resamples a volume to the target dimensions.

    Args:
        vol: The volume to resample, in PyTorch format. Expected shape is [D,H,W]
        target_dimensions: The target dimensions to resample to. Example: (50,256,256) for 50 slices, 256x256 resolution.

    Returns:
        The resampled volume, in PyTorch format.
    """

    assert vol.ndim == 3, "Input volume must be 3D"

    vol = vol.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    resampled_vol = F.interpolate(
        vol,
        size=target_dimensions,
        mode="trilinear",
        align_corners=False,
    )
    return resampled_vol.squeeze(0).squeeze(0)


def resample_batched_tensor(
    x: torch.Tensor, target_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    Fast GPU batch resample:  [B,C,D,H,W] → [B,C,*target_size].
    Runs on the same device as the input tensor.
    """
    assert x.ndim == 5, "expected [B,C,D,H,W]"
    return F.interpolate(
        x,
        size=target_size,
        mode="trilinear",
        align_corners=False,
    )

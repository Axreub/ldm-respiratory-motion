import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import cast
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.ct_views import get_ct_view, view_to_axis
import itertools
import operator
from monai.losses import BendingEnergyLoss, DiffusionLoss


class Sobel3D(nn.Module):
    """3D Sobel filter for edge detection in volumetric data."""

    def __init__(self) -> None:
        super().__init__()
        d = torch.tensor([-1.0, 0.0, 1.0])
        s = torch.tensor([1.0, 2.0, 1.0])

        gx = torch.einsum("i,j,k->ijk", d, s, s)
        gy = torch.einsum("i,j,k->ijk", s, d, s)
        gz = torch.einsum("i,j,k->ijk", s, s, d)
        kernel = torch.stack([gx, gy, gz]).unsqueeze(1)
        self.register_buffer("weight", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = cast(torch.Tensor, self.weight)
        weight = weight.to(dtype=x.dtype, device=x.device)
        return F.conv3d(x, weight, padding=1)


class LPIPS3DLoss(nn.Module):
    """Slice-wise LPIPS implementation for volumetric data."""

    def __init__(self, slice_views: list = ["axial"]) -> None:
        """Initializing type of loss function
        Args:
            slice_views (list): Which plane(s) to use when generating slices.
        """
        super().__init__()
        self.slice_views = sorted(slice_views)
        self.loss_fn = lpips.LPIPS(net="alex").eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, axial_sizes = 0, []
        for view in self.slice_views:
            axial_size = pred.shape[pred.ndim - 3 + view_to_axis(view)]
            axial_sizes.append(axial_size)
            for slice_idx in range(axial_size):
                pred_slice = get_ct_view(pred, view, slice_idx)
                target_slice = get_ct_view(target, view, slice_idx)
                pred_slice = pred_slice.repeat(
                    1, 3, 1, 1
                )  # LPIPS expects 3-channel input
                target_slice = target_slice.repeat(1, 3, 1, 1)
                raw_loss = self.loss_fn(pred_slice, target_slice)
                loss += raw_loss.mean()
        return loss / list(itertools.accumulate(axial_sizes, operator.mul))[-1]


class SSIM3DLoss(nn.Module):
    """3D SSIM for volumetric data."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = StructuralSimilarityIndexMeasure(data_range=2)

    def forward(self, pred, target):
        return 1 - self.loss_fn(pred, target)


class LPELoss(nn.Module):
    """Loss function for autoencoder training, combining L1, Perceptual and Edge-preserving loss (Sobel3D)."""

    def __init__(
        self,
        l1_factor: float = 0.5,
        edge_factor: float = 0.0,
        perceptual_loss_type: str = "lpips",
        slice_views: list = ["axial"],
        variational: bool = False,
    ) -> None:
        """
        Args:
            l1_factor (float, optional): Weight for L1 loss. Defaults to 0.5.
            edge_factor (float, optional): Weight for Sobel loss. Defaults to 0.0.
            perceptual_loss_type (string, optional): Which perceptual loss (if any) to use.
            slice_views (list): Which plane(s) to use when generating slices for perceptual loss.
        """

        super().__init__()

        assert l1_factor + edge_factor <= 1, "l1_factor + edge_factor can not be > 1"
        self.l1_factor = l1_factor
        self.edge_factor = edge_factor
        self.kernel_loss_fn = Sobel3D() if self.edge_factor > 0 else None
        self.l1_loss_fn = nn.L1Loss() if self.l1_factor > 0 else None
        if perceptual_loss_type == "lpips":
            self.perceptual_loss_fn = (
                LPIPS3DLoss(slice_views)
                if self.l1_factor + self.edge_factor < 1
                else None
            )
        elif perceptual_loss_type == "ssim":
            self.perceptual_loss_fn = (
                SSIM3DLoss() if self.l1_factor + self.edge_factor < 1 else None
            )
        else:
            raise ValueError(
                f"loss type must be 'lpips', 'ssim', not '{perceptual_loss_type}'"
            )

    def forward(self, pred_img: torch.Tensor, batch_img: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined L1, Edge and Perceptual loss between predicted and target images. Includes
        """
        loss = 0
        if self.l1_factor > 0:
            loss += self.l1_loss_fn(pred_img, batch_img) * self.l1_factor
        if self.edge_factor > 0:
            edge_maps_pred = self.kernel_loss_fn(pred_img)
            edge_maps = self.kernel_loss_fn(batch_img)
            loss += F.l1_loss(edge_maps_pred, edge_maps) * self.edge_factor
        if self.edge_factor + self.l1_factor < 1:
            loss += self.perceptual_loss_fn(pred_img, batch_img) * (
                1 - self.edge_factor - self.l1_factor
            )
        return loss


class DVFLoss(nn.Module):
    """Loss function for DVF autoencoder training."""

    def __init__(
        self,
        l1_factor: float = 0.9,
        diffusion_factor: float = 0.08,
        bending_energy_factor: float = 0.02,
    ) -> None:
        super().__init__()
        self.l1_loss_fn = nn.L1Loss()
        self.diffusion_loss_fn = DiffusionLoss()
        self.bending_energy_loss_fn = BendingEnergyLoss()

        self.l1_factor = l1_factor
        self.diffusion_factor = diffusion_factor
        self.bending_energy_factor = bending_energy_factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.l1_factor * self.l1_loss_fn(pred, target)
            + self.diffusion_factor * self.diffusion_loss_fn(pred)
            + self.bending_energy_factor * self.bending_energy_loss_fn(pred)
        )

import os
import torch
import numpy as np
import torch.nn.functional as F
import json
import airlab as al
from pathlib import Path
import skimage.filters as filters
import cc3d
import cv2

from utils.path_obtainer import get_paths
from data_processing.utils.dp_utils import get_spacing_from_meta
from utils.dvf import torch_to_airlab, save_bspline


def upsample_control_points(
    prev_control_points: torch.Tensor, target_shape: tuple[int, ...]
):
    """
    Upsamples control points from a previous pyramid level to initialize the next level of bspline transform.
    prev_control_points (torch.Tensor): shape: (1, C, nz, ny, nx)
    target_shape (tuple[int,...]): (nz', ny', nx') without batch/chan dims

    Returns:
        torch.Tensor: Upsampled control points. Shape: (1, C, nz', ny', nx')
    """
    return F.interpolate(
        prev_control_points, size=target_shape, mode="trilinear", align_corners=True
    )


def make_body_mask(vol: torch.Tensor) -> torch.Tensor:
    """
    Makes a body mask from a volume. The body's interior is set to 1, and the exterior is set to 0.

    This preprocessing code is courtesy of https://github.com/cyiheng/Dynagan
    """
    device = vol.device
    arr = vol.squeeze().cpu().numpy()
    threshold = filters.threshold_otsu(arr)  # separate air and non-air
    arr = np.where(arr < threshold, -1000, arr)

    threshold_2 = filters.threshold_otsu(
        arr[arr > threshold]
    )  # separate soft-tissue and bone

    arr = np.where(arr > threshold_2, 1000, arr)
    arr = np.where(arr > -1000, 1, arr)  # set soft-tissue and bone to 1

    connectivity = 6
    # Find connected components in the volume; separated by air. Maps the labels to (0, 1, 2, ...)
    labels_out, _ = cc3d.connected_components(
        arr, connectivity=connectivity, return_N=True
    )

    idx, counts = np.unique(labels_out, return_counts=True)

    # index 0 is the background
    idx_tissues = idx[np.argmax(counts[1:]) + 1]
    tissues = np.where(labels_out != idx_tissues, 0, 1)

    # tissues is a binary mask of the body's outline; fill holes (lungs) by each axial slice.
    res = []
    for i in range(tissues.shape[0]):
        s = tissues[i, :, :]
        s = np.uint8(s)
        im_floodfill = s.copy()

        h, w = s.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = s | im_floodfill_inv
        im_out = np.where(im_out == 255, 1, im_out)
        res += [im_out]

    filled = np.array(res)
    return (
        torch.as_tensor(filled, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def generate_bspline(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    sigma_levels: list[tuple[int, int, int]] = [(8, 8, 8), (4, 4, 4), (2, 2, 2)],
    iter_levels: list[int] = [200, 100, 100],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    regulariser_weights: list[float] = [3e-3, 1e-3, 1e-4],
    learning_rates: list[float] = [5e-3, 3e-3, 1e-3],
    pyramid: list[list[int]] = [[2, 4, 4], [1, 2, 2]],
    bspline_order: int = 1,
) -> tuple["al.transformation.pairwise.BsplineTransformation", torch.Tensor]:
    """
    Uses GPU acceleration to generate a B-spline transform.
    Optimizes the B-spline transform using PyTorch; Adam is used as the optimizer.
    Uses anisotropic TV regularization.
    Optimizes the B-spline transform using a series of pyramid levels. The pyramid levels are created by downsampling the images and masks.
    The B-spline transform is optimized for each pyramid level, and the results are upsampled to the original image size. Each subsequent pyramid level
    uses the previous (upsampled) transform as the initialization for the next level.

    Args:
    fixed (torch.Tensor): Fixed image (The higher phase)
    moving (torch.Tensor): Moving image (The lower phase)
    sigma_levels (list[tuple[int, int, int]]): Specifies how many control points are used (each sigma pixels). Lower sigma means more control points.
    iter_levels (list[int]): Number of iterations for each pyramid level.
    device (torch.device): Device to use. Defaults to cuda if available.
    regulariser_weights (list[float]): Regularisation weights for each pyramid level.
    learning_rates (list[float]): Learning rates for each pyramid level.
    pyramid (list[list[int]]): Specifies the pyramid levels. Each pyramid level downsizes (Z,Y,X) by the factors. E.g. pyramid [2,4,4]
    gives [50, 256, 256] -> [25, 64, 64].
    bspline_order (int): Order of the B-spline. Default is 1 for linear B-splines; 3 for cubic B-splines.

    Returns:
    bspline_transform (al.transformation.pairwise.BsplineTransformation): The B-spline transform in Airlab format.
    moving_mask (torch.Tensor): The moving mask as a torch tensor.
    """

    fixed_mask = make_body_mask(fixed.image)
    moving_mask = make_body_mask(moving.image)

    fixed_mask = torch_to_airlab(fixed_mask, spacing=fixed.spacing, device=device)
    moving_mask = torch_to_airlab(moving_mask, spacing=moving.spacing, device=device)

    fixed_levels = al.create_image_pyramid(fixed, pyramid) if pyramid else [fixed]
    moving_levels = al.create_image_pyramid(moving, pyramid) if pyramid else [moving]
    fixed_mask_levels = (
        al.create_image_pyramid(fixed_mask, pyramid) if pyramid else [fixed_mask]
    )
    moving_mask_levels = (
        al.create_image_pyramid(moving_mask, pyramid) if pyramid else [moving_mask]
    )

    prev_transform = None
    for lvl, (f_lvl, f_mask_lvl, m_lvl, m_mask_lvl) in enumerate(
        zip(fixed_levels, fixed_mask_levels, moving_levels, moving_mask_levels)
    ):
        reg = al.PairwiseRegistration(verbose=False)
        bspline_transform = al.transformation.pairwise.BsplineTransformation(
            m_lvl.size,
            sigma=sigma_levels[lvl],
            order=bspline_order,
            diffeomorphic=False,
            dtype=torch.float32,
            device=device,
        )

        if prev_transform is not None:
            with torch.no_grad():
                src_grid = prev_transform.trans_parameters.detach()
                src_grid = upsample_control_points(
                    src_grid, bspline_transform.trans_parameters.shape[2:]
                )
                bspline_transform.trans_parameters.copy_(src_grid)

        regulariser = al.regulariser.displacement.TVRegulariser(
            pixel_spacing=m_lvl.spacing
        )
        regulariser.set_weight(regulariser_weights[lvl])
        reg.set_regulariser_displacement([regulariser])

        reg.set_transformation(bspline_transform)

        img_loss = al.loss.pairwise.NCC(f_lvl, m_lvl, f_mask_lvl, m_mask_lvl)
        reg.set_image_loss([img_loss])

        optim = torch.optim.Adam(
            bspline_transform.parameters(), lr=learning_rates[lvl], amsgrad=True
        )
        reg.set_optimizer(optim)
        reg.set_number_of_iterations(iter_levels[lvl])
        reg.start()

        prev_transform = bspline_transform

    # Get tensor
    moving_mask = moving_mask.image

    return bspline_transform, moving_mask


def process_phase_pair(
    current_path: str,
    next_path: str,
    sigma_levels: list[tuple[int, int, int]] = [(8, 8, 8), (4, 4, 4), (2, 2, 2)],
    iter_levels: list[int] = [200, 100, 50],
    regulariser_weights: list[float] = [3e-3, 1e-3, 1e-4],
    learning_rates: list[float] = [5e-4, 3e-4, 1e-4],
    pyramid: list[list[int]] = [[2, 4, 4], [1, 2, 2]],
    bspline_order: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Process a single pair of consecutive phases."""

    # Load and convert images
    moving = torch.load(os.path.join(current_path, "volume.pt"))
    fixed = torch.load(os.path.join(next_path, "volume.pt"))

    meta = json.load(open(os.path.join(current_path, "scan_info.json")))

    # airlab expects spacing in (x,y,z) order
    spacing = get_spacing_from_meta(meta, resampled=True)[::-1]

    moving_al = torch_to_airlab(moving, spacing=spacing, device=device)
    fixed_al = torch_to_airlab(fixed, spacing=spacing, device=device)

    bspline_transform, moving_mask = generate_bspline(
        fixed_al,
        moving_al,
        sigma_levels=sigma_levels,
        iter_levels=iter_levels,
        regulariser_weights=regulariser_weights,
        learning_rates=learning_rates,
        pyramid=pyramid,
        bspline_order=bspline_order,
    )
    save_bspline(
        bspline_transform,
        os.path.join(current_path, "bspline_transform.pt"),
        order=bspline_order,
    )
    torch.save(moving_mask, os.path.join(current_path, "mask.pt"))


if __name__ == "__main__":
    from dotenv import load_dotenv
    from tqdm import tqdm
    import warnings

    # Ignore warning from airlab about corner alignment
    warnings.filterwarnings("ignore", message=".*align_corners.*")

    load_dotenv()

    BSPLINE_ORDER = 3
    SIGMA_LEVELS = [(8, 8, 8), (4, 4, 4), (2, 2, 2)]
    ITER_LEVELS = [300, 200, 100]
    REGULARISER_WEIGHTS = [5e-3, 3e-3, 1e-3]
    LEARNING_RATES = [5e-3, 3e-3, 1e-3]
    PYRAMID = [[2, 4, 4], [1, 2, 2]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(get_paths("data_processing")["idc_downloads"]["processed"])
    study_dirs = [
        study
        for patient in root.iterdir()
        if patient.is_dir()
        for study in patient.iterdir()
        if study.is_dir()
    ]

    def count_pairs(study: Path) -> int:
        n = len([f for f in study.iterdir() if f.is_dir()])  # only phase folders
        return max(n - 1, 0)

    total_pairs = sum(map(count_pairs, study_dirs))

    with tqdm(total=total_pairs, desc="Processing phase pairs") as pbar:
        for study in sorted(study_dirs):
            folders = sorted([f for f in study.iterdir() if f.is_dir()])

            for current, next in zip(folders, folders[1:]):  # (0,10%), (10%, 20%), ...
                process_phase_pair(
                    current,
                    next,
                    SIGMA_LEVELS,
                    ITER_LEVELS,
                    REGULARISER_WEIGHTS,
                    LEARNING_RATES,
                    PYRAMID,
                    BSPLINE_ORDER,
                    device,
                )
                pbar.update()

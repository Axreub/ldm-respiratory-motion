from __future__ import annotations

import argparse
import re
from itertools import islice
from pathlib import Path
from typing import Iterable, Tuple

import imageio.v3 as iio
import torch
from tqdm import tqdm
import numpy as np

from data_processing.data_loaders import PairedDataset, get_file_paths
from data_processing.utils.dp_utils import get_spacing_from_meta
from visualization.overlays import DVFOverlay
from utils.path_obtainer import get_paths
from inference.latent_diffusion_inference import LatentDiffusionGenerator
from utils.dvf import warp_image

VIEWS: Tuple[str, ...] = ("axial", "coronal", "sagittal")
OVERLAY_OUT = Path("overlays")
DIFFMAP_OUT = Path("difference_maps")
GIF_OUT = Path("gifs")
PHASE_RE = re.compile(r"series_(\d{1,3}(?:\.\d+)?)%$")

overlay = DVFOverlay(max_arrows=1600, scale=0.005, width=0.003, alpha=0.7)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render DVF overlays / difference maps & GIFs"
    )
    p.add_argument("--patient", type=str, help="patient id to process (e.g. 117)")
    p.add_argument(
        "--num_studies",
        type=int,
        default=1,
        help="limit to first N studies in each patient",
    )
    p.add_argument(
        "--num_slices", type=int, default=1, help="slices per view to render"
    )
    p.add_argument(
        "--start_from",
        type=str,
        default="middle",
        help="start from middle, first, or last slice",
    )
    p.add_argument("--fps", type=int, default=1, help="GIF frames per second")
    p.add_argument(
        "--heatmap", action="store_true", help="use heatmap instead of arrows"
    )
    return p.parse_args()


def list_studies(
    root: Path, patient_filter: str | None, n_studies: int | None
) -> Iterable[Path]:
    """Yield study directories respecting patient / study limits."""
    for patient in sorted(root.glob("patient_*")):
        if patient_filter and patient.name.split("/")[-1] != patient_filter:
            continue
        studies = sorted(patient.glob("study_*"))
        yield from (islice(studies, n_studies) if n_studies else studies)


def make_paired_dataset(study_dir: Path) -> PairedDataset:
    """Return a PairedDataset built from every .pt file under the study directory."""
    paths = get_file_paths([str(study_dir)])
    return PairedDataset(paths, dvf_scale_factor=10)


def series_name(path: Path) -> str:
    """Return directory name like 'series_10.0%'."""
    return path.parent.name


def render_series_pair(
    moving_vol: torch.Tensor,
    dvf: torch.Tensor,
    fixed_vol: torch.Tensor,
    spacing: Tuple[float, float, float],
    out_root: Path,
    num_slices: int,
    start_from: str,
    use_arrows: bool,
) -> None:
    """Wrapper around DVFOverlay calls for one (moving, fixed) pair."""
    overlay.render_dvf_overlay(
        out_dir=out_root,
        vol=moving_vol,
        dvf=dvf,
        views=VIEWS,
        num_slices=num_slices,
        start_from=start_from,
        spacing_mm=spacing,
        use_arrows=use_arrows,
    )
    overlay.render_difference_map(
        out_dir=DIFFMAP_OUT / out_root.relative_to(OVERLAY_OUT),
        fixed_vol=fixed_vol,
        moving_vol=moving_vol,
        dvf=dvf,
        views=VIEWS,
        num_slices=num_slices,
        start_from=start_from,
        spacing_mm=spacing,
    )


IDX_RE = re.compile(r"_(?:idx_?(\d+)|slice_(\d+))_")


def save_gifs(
    study_output_dir: Path,
    fps: int,
    gif_type: str,
) -> None:
    """
    Assemble PNG frames into per-slice GIFs and drop them in
        gifs/overlays/...  or  gifs/difference_maps/...
    """

    # look only for the relevant suffix
    suffix = "_overlay.png" if gif_type == "overlay" else "_diffmap.png"
    base_dir = OVERLAY_OUT if gif_type == "overlay" else DIFFMAP_OUT
    dest_root = GIF_OUT / Path(gif_type + "s")

    for view in VIEWS:
        frames_by_idx: dict[str, list[Path]] = {}

        for png in study_output_dir.glob(f"**/{view}_*{suffix}"):
            m = IDX_RE.search(png.name)
            if m:
                idx = m.group(1) or m.group(2)  # whichever captured
                frames_by_idx.setdefault(idx, []).append(png)
        for idx, pngs in frames_by_idx.items():
            gif_frames = []
            for p in sorted(pngs):
                frame = iio.imread(p)
                gif_frames.append(frame)

            rel_path = study_output_dir.relative_to(base_dir) / f"{view}_idx{idx}.gif"
            gif_path = dest_root / rel_path
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(gif_path, gif_frames, duration=int(1000 / fps), loop=0)


def process_study(
    study_dir: Path, num_slices: int, fps: int, use_arrows: bool, start_from: str
) -> None:
    """Process a study directory by rendering overlays and difference maps for each (moving, fixed) pair and making GIFs."""

    dataset = make_paired_dataset(study_dir)
    inference = LatentDiffusionGenerator()
    diffusion_vol, dvf_shape, _ = dataset[0]
    initial_vol, _, _ = dataset[0]
    cum_dvf = torch.zeros_like(dvf_shape)

    for i in range(len(dataset) - 1):
        moving_vol, dvf, phase = dataset[i]
        phase, moving_vol = phase.to(inference.device).unsqueeze(0), moving_vol.to(
            inference.device
        )
        dvf = (
            inference.generate(image=diffusion_vol.to(inference.device), phase=phase)
            .squeeze(0)
            .to(device="cpu")
        )
        cum_dvf += dvf

        fixed_vol_path = dataset.series[i + 1][0]
        fixed_vol = torch.load(fixed_vol_path, map_location="cpu").float()
        moving_vol = moving_vol.squeeze(0)  # (1,D,H,W) -> (D,H,W)

        meta = dataset.get_metadata(i)
        spacing = get_spacing_from_meta(meta, resampled=False)

        # build output sub‑tree e.g. overlays/patient_12/study_3/series_0%_to_10%
        moving_series = series_name(dataset.series[i][0])
        fixed_series = series_name(fixed_vol_path)
        out_root = (
            OVERLAY_OUT
            / study_dir.parent.name
            / study_dir.name
            / f"{moving_series}_to_{fixed_series}"
        )

        render_series_pair(
            initial_vol.squeeze(0),
            cum_dvf,
            fixed_vol,
            spacing,
            out_root,
            num_slices,
            start_from,
            use_arrows,
        )

        diffusion_vol = warp_image(initial_vol, cum_dvf, spacing, device="cpu").squeeze(
            0
        )

    save_gifs(
        OVERLAY_OUT / study_dir.parent.name / study_dir.name, fps, gif_type="overlay"
    )
    save_gifs(
        DIFFMAP_OUT / study_dir.parent.name / study_dir.name,
        fps,
        gif_type="difference_map",
    )


def format_patient_arg(patient_num: str) -> str:
    """Format patient argument to match the expected structure"""
    return f"patient_{patient_num}_HM10395"


def main() -> None:
    root = Path(get_paths("data_processing")["idc_downloads"]["processed"])
    args = parse_args()
    args.patient = format_patient_arg(args.patient)
    OVERLAY_OUT.mkdir(exist_ok=True)
    DIFFMAP_OUT.mkdir(exist_ok=True)
    GIF_OUT.mkdir(exist_ok=True)

    studies = list(list_studies(root, args.patient, args.num_studies))
    print(f"Processing {len(studies)} studies")
    for study in tqdm(studies, desc="Processing studies"):
        process_study(
            study, args.num_slices, args.fps, not args.heatmap, args.start_from
        )

    print("\nFinished. Browse results in:")
    print(f"  {OVERLAY_OUT}\n  {DIFFMAP_OUT}\n  {GIF_OUT}")


if __name__ == "__main__":
    main()

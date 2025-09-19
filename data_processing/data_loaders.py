import torch
import json
from torch.utils.data import Dataset
from pathlib import Path

from data_processing.utils.dp_utils import (
    split_patient_folders,
    get_file_paths,
    create_dataloaders,
    get_series_phase,
)
from utils.dvf import load_bspline, get_dvf_from_bspline
from args.classes import DiffusionTrainArgs, EncoderTrainArgs
from typing import Union, Optional


class SimpleDataset(Dataset):
    """
    A simple dataset that loads images from a list of file paths.
    """

    def __init__(
        self,
        input_paths: list,
    ):
        self.input_paths = sorted([Path(p) for p in input_paths])

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, idx: int):
        image = torch.load(self.input_paths[idx])
        if image.dim() == 3:
            image = image.unsqueeze(dim=0)
        return image

    def get_metadata(self, idx: int) -> dict:
        """Gets the metadata for a single series."""
        with open(self.input_paths[idx].with_name("scan_info.json")) as f:
            return json.load(f)


class TransformDataset(Dataset):
    """
    A dataset that loads bspline transforms and anatomy masks from a list of file paths, generates
    DVFs from them, and returns those DVFs.
    """

    def __init__(self, input_paths: list, dvf_scale_factor: float):
        self.input_paths = sorted([Path(p) for p in input_paths])
        self.dvf_scale_factor = dvf_scale_factor

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, idx: int):
        """Gets a single DVF from the dataset."""
        bspline_path = self.input_paths[idx]
        mask_path = bspline_path.with_name("mask.pt")
        bspline = load_bspline(bspline_path)

        dvf = get_dvf_from_bspline(bspline) * self.dvf_scale_factor
        mask = torch.load(mask_path).to("cpu").squeeze(0).expand_as(dvf)

        dvf = dvf * mask

        return dvf

    def get_metadata(self, idx: int) -> dict:
        """Gets the metadata for a single series."""
        with open(self.input_paths[idx].with_name("scan_info.json")) as f:
            return json.load(f)


class PairedDataset(Dataset):
    """
    Loads (volume, transform, metadata) triples that live in one directory:
        └─ series_…/
           ├─ volume.pt
           ├─ bspline_transform.pt
           ├─ scan_info.json
           └─ mask.pt

    get_metadata_for_series returns the metadata for a single series.
    """

    def __init__(self, input_paths: list, dvf_scale_factor: float) -> None:
        self.input_paths = sorted(
            [
                Path(p)
                for p in input_paths
                if p.endswith("volume.pt") and "90.0%" not in p
            ]
        )
        self.series = []
        self.dvf_scale_factor = dvf_scale_factor

        for vol_path in self.input_paths:
            bspline_path = vol_path.with_name("bspline_transform.pt")
            metadata_path = vol_path.with_name("scan_info.json")
            mask_path = vol_path.with_name("mask.pt")
            phase = get_series_phase(vol_path)

            assert (
                bspline_path.exists() and metadata_path.exists()
            ), f"Missing files for {vol_path.parent}: {bspline_path}, {metadata_path}"
            self.series.append(
                (vol_path, bspline_path, mask_path, phase, metadata_path)
            )

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int):
        """Gets a single volume, masked DVF, and phase triplet from the dataset."""
        volume_path, bspline_path, mask_path, phase, _ = self.series[idx]

        volume = torch.load(volume_path)
        if volume.dim() == 3:
            volume = volume.unsqueeze(dim=0)
        bspline = load_bspline(bspline_path)
        dvf = get_dvf_from_bspline(bspline) * self.dvf_scale_factor

        mask = torch.load(mask_path).to("cpu").squeeze(0).expand_as(dvf)

        dvf = dvf * mask

        return volume, dvf, phase

    def get_metadata(self, idx: int) -> dict:
        """Gets the metadata for a single series."""
        *_, metadata_path = self.series[idx]
        with open(metadata_path) as f:
            return json.load(f)


class LatentDiffusionDataset(Dataset):
    """
    Loads (volume, dvf, metadata) triples that live in one directory:
        └─ series_…/
           ├─ volume.pt
           ├─ dvf.pt
           ├─ scan_info.json
           └─ mask.pt

    get_metadata_for_series returns the metadata for a single series.
    """

    def __init__(self, input_paths: list) -> None:
        self.input_paths = sorted(
            [
                Path(p)
                for p in input_paths
                if p.endswith("volume.pt") and "90.0%" not in p
            ]
        )
        self.series = []

        for vol_path in self.input_paths:
            dvf_path = vol_path.with_name("dvf.pt")
            metadata_path = vol_path.with_name("scan_info.json")
            mask_path = vol_path.with_name("mask.pt")
            phase = get_series_phase(vol_path)

            assert (
                dvf_path.exists() and metadata_path.exists()
            ), f"Missing files for {vol_path.parent}: {dvf_path}, {metadata_path}"
            self.series.append((vol_path, dvf_path, mask_path, phase, metadata_path))

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int, use_mask: bool = False):
        """Gets a single volume, masked DVF, and phase triplet from the dataset."""
        volume_path, dvf_path, mask_path, phase, _ = self.series[idx]

        volume = torch.load(volume_path)
        if volume.dim() == 3:
            volume = volume.unsqueeze(dim=0)
        dvf = torch.load(dvf_path).to("cpu").squeeze(0)
        if use_mask:
            mask = torch.load(mask_path).to("cpu").squeeze(0).expand_as(dvf)
            return volume, dvf, phase, mask

        return volume, dvf, phase

    def get_metadata(self, idx: int) -> dict:
        """Gets the metadata for a single series."""
        *_, metadata_path = self.series[idx]
        with open(metadata_path) as f:
            return json.load(f)


class MedicalDataLoader:
    """
    Splits train, validation, and test based on the different folders
    corresponding to different patients. Returns a DataLoader for each.
    """

    def __init__(
        self,
        train_args: Union[DiffusionTrainArgs, EncoderTrainArgs],
        input_data_path: Optional[str] = None,
        paired: bool = False,
        data_type: Optional[str] = None,
        dvf_scale_factor: float = 1.0,
    ) -> None:
        """
        Initializes the MedicalDataLoader.

        Args:
            train_args (DiffusionTrainArgs | EncoderTrainArgs): Training arguments containing data loading configuration, including input_data_path, batch_size, num_data_loader_workers, train/val/test ratios, transform_type, and data_loader_seed.
            input_data_path (Optional, str): Path to the directory in which the loader will take its data from. If given as none, it will be taken from train_args instead.
            paired (bool, optional): Whether to use paired datasets (volume, transform pairs). If True, transform_type must be specified. Defaults to False.
            data_type (optional , str): In the case of autoencoder loading, specifies which type of data to load. Accepted values: 'volume', 'dvf'.

        """

        if input_data_path is None:
            input_data_path = train_args.input_data_path

        # Split input by patients to avoid information leakage
        (
            input_train_folders,
            input_val_folders,
            input_test_folders,
        ) = split_patient_folders(
            base_data_path=input_data_path,
            train_ratio=train_args.train_ratio,
            val_ratio=train_args.val_ratio,
            test_ratio=train_args.test_ratio,
            seed=train_args.data_loader_seed,
        )

        input_train_paths = get_file_paths(input_train_folders)
        input_val_paths = get_file_paths(input_val_folders)
        input_test_paths = get_file_paths(input_test_folders)

        if paired == True:
            if data_type == "latent":
                self.train_dataset = LatentDiffusionDataset(input_train_paths)
                self.val_dataset = LatentDiffusionDataset(input_val_paths)
                self.test_dataset = LatentDiffusionDataset(input_test_paths)
            else:
                self.train_dataset = PairedDataset(input_train_paths, dvf_scale_factor)
                self.val_dataset = PairedDataset(input_val_paths, dvf_scale_factor)
                self.test_dataset = PairedDataset(input_test_paths, dvf_scale_factor)

        elif isinstance(train_args, EncoderTrainArgs):
            if data_type == "dvf":
                input_train_paths = [
                    p for p in input_train_paths if "bspline_transform" in p
                ]
                input_val_paths = [
                    p for p in input_val_paths if "bspline_transform" in p
                ]
                input_test_paths = [
                    p for p in input_test_paths if "bspline_transform" in p
                ]

                self.train_dataset = TransformDataset(
                    input_train_paths, dvf_scale_factor
                )
                self.val_dataset = TransformDataset(input_val_paths, dvf_scale_factor)
                self.test_dataset = TransformDataset(input_test_paths, dvf_scale_factor)

            elif data_type == "volume":
                input_train_paths = [p for p in input_train_paths if "volume" in p]
                input_val_paths = [p for p in input_val_paths if "volume" in p]
                input_test_paths = [p for p in input_test_paths if "volume" in p]

                self.train_dataset = SimpleDataset(input_train_paths)
                self.val_dataset = SimpleDataset(input_val_paths)
                self.test_dataset = SimpleDataset(input_test_paths)

            else:
                print(
                    "Error: MedicalDataLoader created for autoencoder, but no valid data_type input is given. Accepted: 'volume', 'dvf'."
                )

        self.batch_size = train_args.batch_size
        self.num_workers = train_args.num_data_loader_workers

    def get_dataloaders(
        self,
    ) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """
        Creates and returns DataLoaders for train, validation, and test datasets.

        Returns:
            tuple: A tuple containing train, validation, and test DataLoaders.
        """

        return create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.batch_size,
            self.num_workers,
        )

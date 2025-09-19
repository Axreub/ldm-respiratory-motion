import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import random
import re


def get_series_phase(path: Path | str) -> float:
    """Get the phase of a series from the path and normalize to 0-9 for use in a one-hot encoding."""
    path = str(path)
    PHASE_RE = re.compile(r",_(\d{1,2})\.\d%")
    match = PHASE_RE.search(path)
    if match:
        phase = int(int(match.group(1)) / 10)  # (0 - 90) -> (0 - 9)
        return torch.tensor(phase, dtype=torch.long)
    else:
        return None


def split_patient_folders(
    base_data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Splits a dataset of patient subfolders into train, validation, and test sets.

    This function is designed for medical imaging datasets where each patient's data is stored
    in a separate subfolder under a common parent directory. The split is performed at the
    patient (subfolder) level to prevent data leakage between sets.

    Args:
        base_data_path (str): Path to the parent directory containing all patient subfolders.
        train_ratio (float): Proportion of patients to include in the training set. Defaults to 0.8.
        val_ratio (float): Proportion of patients to include in the validation set. Defaults to 0.1.
        test_ratio (float): Proportion of patients to include in the test set. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[list[str], list[str], list[str]]: Three lists containing the paths to the patient subfolders
            for the training, validation, and test sets, respectively.
    """

    assert (
        train_ratio + val_ratio + test_ratio == 1.0
    ), "Train, validation, and test ratios must sum to 1"

    random.seed(seed)

    input_subfolders = [
        os.path.join(base_data_path, d)
        for d in os.listdir(base_data_path)
        if os.path.isdir(os.path.join(base_data_path, d))
    ]

    input_subfolders = sorted(input_subfolders)
    random.shuffle(input_subfolders)

    num_folders = len(input_subfolders)
    num_train = round(train_ratio * num_folders)
    num_val = round(val_ratio * num_folders)

    train_folders = input_subfolders[:num_train]
    val_folders = input_subfolders[num_train : num_train + num_val]

    # Test becomes the last folders
    test_folders = input_subfolders[num_train + num_val :]

    return train_folders, val_folders, test_folders


def get_file_paths(folders: list[str]) -> list[str]:
    """
    Gets all torch tensors in a given array of folders and their subfolders recursively
    """
    file_paths = []
    for folder in folders:
        # Use rglob to recursively search through all subdirectories
        file_paths.extend(sorted(Path(folder).rglob("*.pt")))
    return [str(path) for path in file_paths]


def get_spacing_from_meta(meta: dict, resampled: bool) -> tuple[float, float, float]:
    """Retrieve pixel spacing [mm] from loaded metadata json in (dz, dy, dx) order."""
    key_prefix = "resampled_" if resampled else ""
    return (
        meta[f"{key_prefix}slice_thickness"],
        meta[f"{key_prefix}pixel_spacing"][1],
        meta[f"{key_prefix}pixel_spacing"][0],
    )


def create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and test sets."""
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True if len(train_dataset) > 0 else False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return train_loader, val_loader, test_loader

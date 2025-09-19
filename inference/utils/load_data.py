import os
import random
import torch
from typing import Optional, Tuple, Union
from torch.utils.data import Subset

from utils.tensor_utils import unsqueeze_channels
from args.classes import DiffusionTrainArgs, EncoderTrainArgs
from data_processing.data_loaders import MedicalDataLoader


def load_dataset(
    train_args: DiffusionTrainArgs | EncoderTrainArgs,
    dataloader_type: str = "val",
    input_data_path: Optional[str] = None,
    data_type: Optional[str] = "volume",
):
    """
    Loads a dataset using the provided training arguments and dataloader type.

    This function constructs the appropriate dataset (train, validation, or test) using the same
    data loading logic as the training process, ensuring consistency and preventing data leakage.
    The returned  is updated to reflect the current storage base path for input file paths.

    Args:
        train_args (DiffusionTrainArgs | EncoderTrainArgs): Training arguments specifying data paths and split ratios.
        dataloader_type (str, optional): Which split to load. Must be one of "train", "val", or "test".
            Defaults to "val".
        input_data_path (Optional, str): Path to the directory in which the loader will take its data from. If given as none, it will be taken from train_args instead.
        data_type(str, optional): In case the dataset is being loaded for an autoencoder, determines whether to load volumes or DVFs.
            Accepted values: None, 'volume', 'dvf'. Defaults to 'volume'.

    Returns:
        torch.utils.data.Dataset: The requested dataset split with updated input file paths.
    """

    if isinstance(train_args, DiffusionTrainArgs):
        paired = True
    elif isinstance(train_args, EncoderTrainArgs):
        paired = False
    else:
        raise ValueError("Invalid train_args input for load_dataset method.")

    train_loader, val_loader, test_loader = MedicalDataLoader(
        train_args, input_data_path, paired, data_type
    ).get_dataloaders()
    if dataloader_type == "val":
        dataloader = val_loader
    elif dataloader_type == "test":
        dataloader = test_loader
    elif dataloader_type == "train":
        dataloader = train_loader
    else:
        raise KeyError("dataloader_type only takes 'val', 'test', or 'train'")

    dataset = dataloader.dataset
    return dataset


def load_samples(
    train_args: DiffusionTrainArgs | EncoderTrainArgs,
    input_data_path: Optional[str] = None,
    dataloader_type: str = "val",
    n_samples: int = 1,
    start_index: int | None = 0,
    data_type: Optional[str] = "volume",
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, list[str]], Tuple[torch.Tensor, list[str]]
]:
    """
    Loads and returns a stack of data samples from a data set in train_args.

    Args:
        train_args (DiffusionTrainArgs | EncoderTrainArgs): Training arguments.
        input_data_path (Optional, str): Path to the directory in which the loader will take its data from. If given as none, it will be taken from train_args instead.
        dataloader_type (str): Type of dataloader to use.
        n_samples (int): amount of samples that will be stacked and loaded.
        start_index (int | None): The index in the data set that the samples will be loaded from. If start_index is None, the samples will be chosen randomly from the dataset.

    Returns:
        If paired (DiffusionTrainArgs):
            Tuple[torch.Tensor, torch.Tensor, list[str]]:
                - stacked_image_tensor: Tensor of shape (n_samples, *image_shape)
                - stacked_dvf_tensor: Tensor of shape (n_samples, *transform_shape)
                - file_paths: List of file paths for the loaded samples
        If unpaired (EncoderTrainArgs):
            Tuple[torch.Tensor, list[str]]:
                - stacked_tensor: Tensor of shape (n_samples, *image_shape)
                - file_paths: List of file paths for the loaded samples
    """

    if isinstance(train_args, DiffusionTrainArgs):
        paired = True
    elif isinstance(train_args, EncoderTrainArgs):
        paired = False
    else:
        raise ValueError("Invalid train_args input for load_samples method.")

    dataset = load_dataset(train_args, dataloader_type, input_data_path, data_type)
    device = "cuda" if torch.cuda.is_available else "cpu"

    if start_index is None:
        indices = random.sample(range(len(dataset)), n_samples)
    else:
        indices = list(range(start_index, start_index + n_samples))
    data_subset = Subset(dataset, indices)
    file_paths = [dataset.input_paths[i] for i in indices]

    tensors = [data_subset[i] for i in range(len(data_subset))]
    if paired:
        images = [
            ensure_channel_dim(series_triplet[0], device) for series_triplet in tensors
        ]
        image_batch = torch.stack(images, dim=0)

        dvfs = [
            ensure_channel_dim(series_triplet[1], device) for series_triplet in tensors
        ]
        dvf_batch = torch.stack(dvfs, dim=0)

        phases = [series_triplet[2].to(device) for series_triplet in tensors]
        phase_batch = torch.stack(phases, dim=0)

        return image_batch, dvf_batch, phase_batch, file_paths
    else:
        stacked_tensor = torch.stack(
            [ensure_channel_dim(tensor, device) for tensor in tensors], dim=0
        )
        return stacked_tensor, file_paths


def ensure_channel_dim(
    tensor: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Ensures that a newly loaded tensor has a channel dimension, and moves it to a set device."""
    if tensor.dim() == 3:
        tensor = unsqueeze_channels(tensor, device)
    else:
        tensor = tensor.to(device)
    return tensor

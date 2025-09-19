import json
import os

from dotenv import load_dotenv
from typing import Union, Tuple
from utils.path_obtainer import get_paths
from datetime import datetime

from args.classes import (
    SamplingArgs,
    DiffusionArgs,
    DiffusionTrainArgs,
    EncoderArgs,
    EncoderTrainArgs,
)

load_dotenv()


def load_args_dict(args_category: str = "diffusion") -> dict:
    """Loads the args file and returns it as a dictionary."""

    BASE_PATH = os.getenv("BASE_PATH")
    args_filepath = os.path.join("args", args_category + "_args.json")
    location_path = os.path.join(BASE_PATH, args_filepath)
    print(f"Fetching args from filename: {location_path}")

    if not os.path.exists(location_path):
        raise FileNotFoundError(f"Args file '{location_path}' not found.")

    with open(location_path, "r") as file:
        args_dict = json.load(file)
    return args_dict


def get_args(
    args_category: str,
) -> Union[
    Tuple[SamplingArgs, DiffusionArgs, DiffusionTrainArgs],
    Tuple[EncoderArgs, EncoderTrainArgs],
]:
    """
    Load and construct argument dataclass objects for the specified model category.

    This function reads the corresponding args JSON configuration file for the given category
    (either 'diffusion' or 'autoencoder'), loads the arguments, and returns them as initialized
    dataclass objects. It also updates relevant path fields using the environment and paths config.

    Args:
        args_category (str): The category of arguments to load. Must be either 'diffusion' or 'autoencoder'.

    Returns:
        Union[
            Tuple[SamplingArgs, DiffusionArgs, DiffusionTrainArgs],
            Tuple[EncoderArgs, EncoderTrainArgs],
        ]:
            - If args_category is 'diffusion': returns (sampling_args, model_args, train_args)
            - If args_category is 'autoencoder': returns (model_args, train_args)
    """

    assert args_category in (
        "diffusion",
        "autoencoder",
    ), f"Error: Invalid args category in get_args: {args_category}"

    args_dict = load_args_dict(args_category)

    model_args_dict = args_dict[args_category + "_args"]
    train_args_dict = args_dict[args_category + "_train_args"]

    paths = get_paths(args_category)

    if args_category == "diffusion":
        sampling_args_dict = args_dict["sampling_args"]

        sampling_args = SamplingArgs(**sampling_args_dict)
        model_args = DiffusionArgs(**model_args_dict)
        train_args = DiffusionTrainArgs(**train_args_dict)
        train_args.input_data_path = paths["input_data_path"]
        train_args.warm_checkpoint_path = paths["warm_checkpoint_path"]
        train_args.save_path = os.path.join(
            paths["save_path"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        return sampling_args, model_args, train_args

    elif args_category == "autoencoder":
        model_args = EncoderArgs(**model_args_dict)
        train_args = EncoderTrainArgs(**train_args_dict)
        train_args.input_data_path = paths["input_data_path"]
        train_args.warm_checkpoint_path = paths["warm_checkpoint_path"]
        train_args.save_path = os.path.join(
            paths["save_path"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        return model_args, train_args

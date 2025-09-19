import torch
import torch.nn as nn
import math
import os
import json

from dataclasses import asdict
from typing import Any

from backbones.model_loss import LPELoss, DVFLoss
from training.utils.decorators import rank_zero
from args.classes import (
    DiffusionArgs,
    DiffusionTrainArgs,
    EncoderArgs,
    EncoderTrainArgs,
)


@rank_zero
def save_args_to_json(
    model_args: DiffusionArgs | EncoderArgs,
    train_args: DiffusionTrainArgs | EncoderTrainArgs,
    args_file_name: str = "args.json",
    save_dir: str = None,
):
    """
    Save the attributes of model_args and train_args as key/value pairs in a single .json file.

    Args:
        model_args: Model argument dataclass (DiffusionArgs or EncoderArgs)
        train_args: Training argument dataclass (DiffusionTrainArgs or EncoderTrainArgs)
        args_file_name: Name of the file to save the arguments to. Defaults to "args.json".
        save_dir: Directory to save the arguments to. Defaults to train_args.save_path.
    """
    if save_dir is None:
        save_dir = train_args.save_path
    model_args_dict = asdict(model_args)
    train_args_dict = asdict(train_args)

    args_dict = {
        "model_args": model_args_dict,
        "train_args": train_args_dict,
    }

    os.makedirs(save_dir, exist_ok=True)
    args_path = os.path.join(save_dir, args_file_name)

    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Saved args to {args_path}")


def load_optim(
    learning_rate: float,
    optimizer: str,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """
    Loads and returns an optimizer for the given model.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Name of the optimizer ('adamw' or 'sgd').
        model (nn.Module): The model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """

    if optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            f"Invalid optimizer: {optimizer}. Supported values are 'adamw' and 'sgd'."
        )
    return optim


def load_lr_scheduler(
    optim: torch.optim.Optimizer,
    cycle_steps: int,
    lr_max: float,
    warmup_steps: int = 0,
    lr_min: float = 0.0,
    schedule_mode: str = "cosine",
    gradient_accumulation_steps: int = 1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Loads and returns a learning rate scheduler.

    Args:
        optim (torch.optim.Optimizer): The optimizer to schedule.
        cycle_steps (int): Total number of steps in a cycle.
        lr_max (float): Maximum learning rate.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 0.
        lr_min (float, optional): Minimum learning rate. Defaults to 0.0.
        schedule_mode (str, optional): Scheduler mode. Only 'cosine' is supported. Defaults to "cosine".
        gradient_accumulation_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.

    Raises:
        ValueError: If an unsupported schedule mode is specified.
    """
    assert warmup_steps >= 0, "Warmup steps must be non-negative"
    if schedule_mode != "cosine":
        raise ValueError(
            f"Invalid schedule mode: {schedule_mode}. Supported: 'cosine'."
        )
    warmup_steps = warmup_steps / gradient_accumulation_steps
    cycle_steps = cycle_steps / gradient_accumulation_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: LR goes from 0 to initial lr provided by optim during warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / max(1, cycle_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            min_max_ratio = lr_min / lr_max
            return min_max_ratio + (1 - min_max_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    return scheduler


def load_loss_fn(
    loss_fn: str,
    **kwargs: Any,
) -> nn.Module:
    """
    Loads and returns a loss function.

    Args:
        loss_fn (str): Name of the loss function ('mse', 'l1', or 'epm').
        **kwargs: Additional keyword arguments for the loss function.

    Returns:
        nn.Module: The loss function.
    """
    train_args = kwargs["kwargs"]
    if loss_fn.lower() == "mse":
        return nn.MSELoss()
    elif loss_fn.lower() == "lpe":
        return LPELoss(
            l1_factor=train_args.l1_factor,
            edge_factor=train_args.edge_factor,
            perceptual_loss_type=train_args.perceptual_loss_type,
            slice_views=train_args.slice_views,
        )
    elif loss_fn.lower() == "l1":
        return nn.L1Loss()
    elif loss_fn.lower() == "dvf":
        return DVFLoss()
    else:
        raise ValueError(
            f"Invalid loss function: {loss_fn}. Supported values are 'mse', 'lpe' and 'l1'."
        )


@rank_zero
def log_run_info(trainer_obj: Any) -> None:
    """
    Logs run parameters and data loader info from a trainer object.
    Expects a logger, diffusion or encoder args and a data loader.

    Args:
        trainer_obj (DiffusionTrianer | EncoderTrainer): The trainer object containing logger, model, train_args, and data loaders.

    Returns:
        None
    """
    logger = trainer_obj.logger

    logger.info("Start of Training Run with following parameters:")
    logger.info("")  # Add newline

    logger.info("Training parameters:")
    for key, value in asdict(trainer_obj.train_args).items():
        logger.info("  {}: {}".format(key, value))
    logger.info("")

    def _format_number(num: int) -> str:
        """Helper function for easy param num readability."""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)

    logger.info("Number of model parameters:")
    total_params = sum(p.numel() for p in trainer_obj.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer_obj.model.parameters() if p.requires_grad
    )

    logger.info(
        "  Total model parameters: {} ({})".format(
            total_params, _format_number(total_params)
        )
    )
    logger.info(
        "  Trainable model parameters: {} ({})".format(
            trainable_params, _format_number(trainable_params)
        )
    )

    if hasattr(trainer_obj, "diffusion_args"):
        logger.info("Diffusion parameters:")
        for key, value in asdict(trainer_obj.diffusion_args).items():
            logger.info("  {}: {}".format(key, value))

    elif hasattr(trainer_obj, "encoder_args"):
        logger.info("Encoder parameters:")
        for key, value in asdict(trainer_obj.encoder_args).items():
            logger.info("  {}: {}".format(key, value))
    logger.info("")

    num_training_batches = len(trainer_obj.train_loader)
    num_training_samples = len(trainer_obj.train_loader.dataset)
    logger.info(
        "Training data loader: {} batches ({} samples total).".format(
            num_training_batches, num_training_samples
        )
    )

    num_validation_batches = len(trainer_obj.val_loader)
    num_validation_samples = len(trainer_obj.val_loader.dataset)
    logger.info(
        "Validation data loader: {} batches ({} samples total).".format(
            num_validation_batches, num_validation_samples
        )
    )

    num_test_batches = len(trainer_obj.test_loader)
    num_test_samples = len(trainer_obj.test_loader.dataset)
    logger.info(
        "Test data loader: {} batches ({} samples total).".format(
            num_test_batches, num_test_samples
        )
    )
    logger.info("")


def make_deterministic(seed: int = 100, use_benchmark: bool = False) -> None:
    """
    Sets random seeds and backend flags for deterministic training.

    Args:
        seed (int, optional): The random seed to use. Defaults to 100.
        use_benchmark (bool, optional): Whether to enable cudnn.benchmark. Defaults to False.

    Returns:
        None
    """
    import torch
    import random
    import numpy as np
    import os

    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = use_benchmark

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

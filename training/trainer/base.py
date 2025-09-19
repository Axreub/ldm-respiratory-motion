import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import os
import numpy as np
from dataclasses import asdict
import matplotlib.pyplot as plt
from typing import Any, Callable, Union, Optional
from args.classes import (
    DiffusionTrainArgs,
    EncoderTrainArgs,
)
from torch.utils.data import DataLoader

from training.utils.train_utils import (
    load_optim,
    load_loss_fn,
    load_lr_scheduler,
)
from utils.setup_logging import setup_training_logging
from training.utils.decorators import rank_zero


class Trainer:
    """
    The base class for training. Implements model saving, most initialization, logging helper functions
    Extend it with a training loop to create a fully operational model trainer
    """

    def __init__(
        self,
        model: Any,
        train_args: Union[DiffusionTrainArgs, EncoderTrainArgs],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Union[str, torch.device] = "cpu",
        variational: Optional[bool] = False,
    ):
        """
        Initializes the Trainer base class.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_args (DiffusionTrainArgs | EncoderTrainArgs): Training configuration arguments.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
            device (str | torch.device): The device to use for training. Default is "cuda".
        Returns:
            None.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_args = train_args
        self.is_distributed = isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        )
        self.variational = variational

        for arg_name, value in asdict(train_args).items():
            setattr(self, arg_name, value)

        self.device = device

        self.optim = load_optim(self.lr_max, self.optimizer, model)
        self.loss_fn = load_loss_fn(self.loss_fn, kwargs=train_args).to(
            device=self.device
        )

        if self.lr_scheduler is not None:
            total_steps = self.n_epochs * len(self.train_loader)
            self.scheduler = load_lr_scheduler(
                self.optim,
                total_steps,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
                schedule_mode=self.lr_scheduler,
                warmup_steps=self.warmup_steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
        else:
            self.scheduler = None

        self.scaler = GradScaler(device=self.device)

        self.logger = setup_training_logging(logfile_path=self.save_path)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.save_path, "logs/tensorboard_logs")
        )

        self.training_losses = []
        self.validation_losses = []

    @rank_zero
    def log(self, msg: str, level: str = "info", *args, **kwargs) -> None:
        """Log a message using the configured logger, but only on the main process."""
        getattr(self.logger, level)(msg, *args, **kwargs)

    def _get_device_type(self) -> str:
        """Returns the device type as a string ("cuda" or "cpu") based on self.device (for autocast)."""

        if isinstance(self.device, str):
            device_type = "cuda" if self.device == "cuda" else "cpu"
        else:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
        return device_type

    @rank_zero
    def _log_train_loss(self, total_iters: int) -> None:
        """Logs training loss on the main process."""

        self.writer.add_scalar(
            "Loss/train",
            np.array(self.training_losses[-1]),
            global_step=total_iters,
        )
        self.writer.flush()

    @rank_zero
    def save_model(self, training_iters: int) -> None:
        """Save the current model state, training arguments, and model arguments to disk."""

        try:
            f = os.path.join(self.save_path, f"{training_iters}iter.pt")

            torch.save(
                (
                    self.model.module.state_dict()
                    if self.is_distributed
                    else self.model.state_dict()
                ),
                f,
            )
            self.log(f"Saved model at iteration {training_iters} to {f}")
        except Exception as e:
            self.log(
                f"Failed to save model at iteration {training_iters}: {e}",
                level="error",
                exc_info=True,
            )
            raise

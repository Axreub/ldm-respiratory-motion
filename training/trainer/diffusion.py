import torch
import os
import time
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from diffusion.utils.noise_schedules import get_noise_schedule
from args.classes import DiffusionTrainArgs
from torch.utils.data import DataLoader
from utils.tensor_utils import unsqueeze_channels
from typing import Any, Tuple, List, Union, Optional

from inference.utils.load_models import load_autoencoder

from training.utils.train_utils import log_run_info
from training.utils.decorators import rank_zero
from training.trainer.base import Trainer
from data_processing.dicom.normalizer import resample_batched_tensor


class DiffusionTrainer(Trainer):
    """
    Trainer object for diffusion model. Expects train/val/test splitting
    """

    def __init__(
        self,
        model: Any,
        train_args: DiffusionTrainArgs,
        noise_schedule: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            model, train_args, train_loader, val_loader, test_loader, device
        )

        beta = get_noise_schedule(self.T, noise_schedule)

        self.alpha = 1.0 - beta
        self.ps_alpha = torch.cumprod(self.alpha, dim=0)
        self.alpha = self.alpha.to(self.device)
        self.ps_alpha = self.ps_alpha.to(self.device)

        # Initialize evaluator
        self.evaluator = DiffusionEvaluator(self)

    def perturb_image(self, x, t, noise):
        try:
            t = t.view(-1, *([1] * (x.dim() - 1)))
            ps_alpha_t = self.ps_alpha[t - 1]
            return torch.sqrt(ps_alpha_t) * x + torch.sqrt((1 - ps_alpha_t)) * noise
        except Exception as e:
            self.log(f"Error while perturbing image: {e}", level="error", exc_info=True)
            raise

    def evaluate(self, total_iters, epoch, dataset_type="validation"):
        """
        Evaluate the model using fixed time embeddings.

        dataset_type: "validation" or "test"
        """

        self.evaluator.evaluate(total_iters, epoch, dataset_type)

    def training_loop(self):
        """The main training loop script for the diffusion model."""
        start_time = time.time()
        total_iters = 0
        mean_loss = 0

        log_run_info(trainer_obj=self)

        try:
            gradient_accumulation_progress = 0
            for epoch in range(1, self.n_epochs):
                epoch_start_time = time.time()
                self.train_loader.sampler.set_epoch(epoch)

                progress_bar = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc="Training Progress",
                    dynamic_ncols=True,
                )

                for batch_index, (image_batch, dvf_batch, phase_batch) in progress_bar:
                    try:
                        if self.resample_target_dim is not None:
                            image_batch = resample_batched_tensor(
                                image_batch, self.resample_target_dim
                            )
                            dvf_batch = resample_batched_tensor(
                                dvf_batch, self.resample_target_dim
                            )

                        dvf_batch = dvf_batch.to(self.device)

                        image_batch = image_batch.to(self.device)

                        t = torch.randint(
                            1, self.T, (dvf_batch.shape[0],), device=self.device
                        )
                        dvf_batch_noise = torch.randn_like(
                            dvf_batch, device=self.device
                        )
                        perturbed_dvf_batch = self.perturb_image(
                            dvf_batch, t, dvf_batch_noise
                        )

                        with autocast(
                            device_type=self._get_device_type(),
                            dtype=torch.float16,
                        ):
                            concat_batch = torch.cat(
                                (image_batch, perturbed_dvf_batch), dim=-4
                            )  # Concatenate volumes and DVFs, add volumes as context for the diffusion mode
                            pred_noise = self.model(
                                concat_batch, t / self.T, phase_batch
                            )
                            loss = self.loss_fn(pred_noise, dvf_batch_noise)
                            scaled_loss = loss / self.gradient_accumulation_steps
                            mean_loss += loss.item() / self.gradient_accumulation_steps

                        self.scaler.scale(scaled_loss).backward()
                        gradient_accumulation_progress += 1
                        if (
                            gradient_accumulation_progress + 1
                        ) % self.gradient_accumulation_steps == 0:
                            self.scaler.step(self.optim)
                            self.scaler.update()
                            self.optim.zero_grad()

                            gradient_accumulation_progress = 0
                            total_iters += 1

                            if self.scheduler is not None:
                                self.scheduler.step()

                            self.training_losses.append(mean_loss)

                            progress_bar.set_postfix(loss=f"{mean_loss:.4f}")
                            mean_loss = 0

                            with torch.no_grad():
                                self._log_train_loss(total_iters)

                                if total_iters % self.save_iters == 0:
                                    self.save_model(total_iters)

                    except Exception as batch_e:
                        self.log(
                            f"Error processing batch {batch_index} in epoch {epoch}: {batch_e}",
                            level="error",
                            exc_info=True,
                        )
                        continue

                # Compute validation loss and log
                if len(self.val_loader) > 0:
                    self.evaluate(total_iters=total_iters, epoch=epoch)
                self.log(
                    f"Epoch {epoch} completed, time spent: {time.time()-epoch_start_time:.2f}s"
                )

            with torch.no_grad():
                self.save_model(total_iters)

            # Get test loss at training end
            if len(self.test_loader) > 0:
                self.evaluate(total_iters=total_iters, epoch=epoch, dataset_type="test")
            self.log(
                f"Training completed, total time spent: {time.time()-start_time:.2f}s"
            )

        except Exception as e:
            self.log(f"Error in training loop: {e}", level="error", exc_info=True)
            raise


class DiffusionEvaluator:
    """
    Handles evaluation logic for diffusion models.

    Args:
        trainer (DiffusionTrainer): The trainer object containing model, dataloaders, and configuration.
    """

    def __init__(self, trainer: "DiffusionTrainer") -> None:
        """
        Initialize the DiffusionEvaluator.

        Args:
            trainer (DiffusionTrainer): The trainer object to evaluate.
        """
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device
        self.T = trainer.T
        self.ps_alpha = trainer.ps_alpha
        self.writer = trainer.writer
        self.perturb_image = trainer.perturb_image

    def _get_eval_config(
        self, dataset_type: str
    ) -> Tuple[DataLoader, torch.Tensor, Optional[str]]:
        """
        Get evaluation configuration for validation or test.

        Args:
            dataset_type (str): Either "validation" or "test".

        Returns:
            Tuple[DataLoader, torch.Tensor, Optional[str]]:
                - DataLoader for the specified dataset type.
                - Fixed time steps tensor for evaluation.
                - TensorBoard tag for logging (None for test).

        Raises:
            ValueError: If dataset_type is not "validation" or "test".
        """
        if dataset_type == "validation":
            loader = self.trainer.val_loader
            if not hasattr(self.trainer, "fixed_validation_ts"):
                self.trainer.fixed_validation_ts = torch.randint(
                    1,
                    self.T,
                    (len(loader) * self.trainer.batch_size,),
                    device=self.device,
                )
            fixed_ts = self.trainer.fixed_validation_ts
            tensorboard_tag = "Loss/validation"
        elif dataset_type == "test":
            loader = self.trainer.test_loader
            if not hasattr(self.trainer, "fixed_test_ts"):
                self.trainer.fixed_test_ts = torch.randint(
                    1,
                    self.T,
                    (len(loader) * self.trainer.batch_size,),
                    device=self.device,
                )
            fixed_ts = self.trainer.fixed_test_ts
            tensorboard_tag = None
        else:
            raise ValueError("dataset_type must be 'validation' or 'test'.")

        return loader, fixed_ts, tensorboard_tag

    def evaluate(
        self,
        total_iters: int,
        epoch: int,
        dataset_type: str = "validation",
    ) -> None:
        """
        Evaluate the model using fixed time embeddings.

        Args:
            total_iters (int): Number of training iterations completed.
            epoch (int): Current epoch number.
            dataset_type (str, optional): Dataset type ("validation" or "test"). Defaults to "validation".

        Returns:
            None
        """
        self.model.eval()

        with torch.no_grad():
            loader, fixed_ts, tensorboard_tag = self._get_eval_config(dataset_type)
            losses = []
            sample_idx = 0
            accumulated_loss = 0.0

            progress_bar = tqdm(
                enumerate(loader),
                total=len(loader),
                desc="Evaluation Progress",
                dynamic_ncols=True,
            )

            for batch_index, (image_batch, dvf_batch, phase_batch) in progress_bar:
                try:
                    image_batch = image_batch.to(self.device)
                    dvf_batch = dvf_batch.to(self.device)

                    if dvf_batch.dim() == 4:
                        dvf_batch = unsqueeze_channels(dvf_batch, device=self.device)
                    if image_batch.dim() == 4:
                        image_batch = unsqueeze_channels(
                            image_batch, device=self.device
                        )

                    batch_noise = torch.randn_like(dvf_batch, device=self.device)

                    batch_size = dvf_batch.shape[0]
                    t = fixed_ts[sample_idx : sample_idx + batch_size]
                    sample_idx += batch_size

                    perturbed_dvf_batch = self.perturb_image(dvf_batch, t, batch_noise)

                    concat_batch = torch.cat((image_batch, perturbed_dvf_batch), dim=-4)

                    with autocast(
                        device_type=self.trainer._get_device_type(), dtype=torch.float16
                    ):
                        pred_noise = self.model(concat_batch, t / self.T, phase_batch)
                        loss = self.trainer.loss_fn(pred_noise, batch_noise)
                        scaled_loss = loss / self.trainer.gradient_accumulation_steps
                    accumulated_loss += scaled_loss.item()

                    if (
                        batch_index + 1
                    ) % self.trainer.gradient_accumulation_steps == 0:
                        losses.append(accumulated_loss)
                        progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}")
                        accumulated_loss = 0.0

                except Exception as e:
                    self.trainer.log(
                        f"Error processing {dataset_type} batch {batch_index} in epoch {epoch}: {e}",
                        level="error",
                        exc_info=True,
                    )
                    continue

            self._log_results(losses, total_iters, dataset_type, tensorboard_tag)

        self.model.train()

    @rank_zero
    def _log_results(
        self,
        losses: List[float],
        total_iters: int,
        dataset_type: str,
        tensorboard_tag: Optional[str],
    ) -> None:
        """
        Log evaluation results.

        Args:
            losses (List[float]): List of loss values for each batch.
            total_iters (int): Number of training iterations completed.
            dataset_type (str): Dataset type ("validation" or "test").
            tensorboard_tag (Optional[str]): Tag for TensorBoard logging.

        Returns:
            None
        """
        avg_loss = np.mean(losses) if losses else float("nan")

        if tensorboard_tag is not None:
            self.writer.add_scalar(tensorboard_tag, avg_loss, global_step=total_iters)

        self.trainer.log(
            f"Iteration {total_iters}, {dataset_type} loss: {avg_loss:.4f}"
        )

        if dataset_type == "validation":
            self.trainer.validation_losses.extend(losses)
            np.save(
                os.path.join(self.trainer.save_path, "validation_loss.npy"),
                np.array(self.trainer.validation_losses),
            )

import torch
import os
import time
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
import torch.distributed as dist

from utils.tensor_utils import unsqueeze_channels
from training.utils.train_utils import log_run_info
from training.trainer.base import Trainer


class EncoderTrainer(Trainer):
    """
    Trainer object for autoencoder. Expects train/val/test splitting.
    """

    def evaluate(self, total_iters: int, epoch: int) -> None:
        """
        Performs model evaluation on the validation data set and logs results.

        Args:
            total_iters (int): The amount of training iters performed at the time when validation is started.
            epoch (int): The training epoch during which evaluation is started.

        Returns:
            None.
        """
        self.model.eval()

        with torch.no_grad():
            validation_losses = []
            accumulated_loss = 0.0

            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validation Progress",
                dynamic_ncols=True,
            )

            for batch_index, batch in progress_bar:
                try:
                    batch = batch.to(self.device)
                    if batch.dim() == 4:
                        batch = unsqueeze_channels(batch, device=self.device)

                    with autocast(device_type=self._get_device_type()):
                        output = self.model(batch)
                        loss = self.loss_fn(output, batch)
                        scaled_loss = loss / self.gradient_accumulation_steps
                    accumulated_loss += scaled_loss.item()

                    if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                        validation_losses.append(accumulated_loss)
                        progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}")
                        accumulated_loss = 0.0

                except Exception as batch_e:
                    self.log(
                        f"Error processing validation batch {batch_index} in epoch {epoch}: {batch_e}",
                        level="error",
                        exc_info=True,
                    )
                    continue

            avg_val_loss = (
                np.mean(validation_losses) if validation_losses else float("nan")
            )
            if not self.is_distributed or dist.get_rank() == 0:
                self.writer.add_scalar(
                    "Loss/validation",
                    avg_val_loss,
                    global_step=total_iters,
                )
            self.log(f"Iteration {total_iters}, validation loss: {avg_val_loss:.4f}")

            self.validation_losses.append(avg_val_loss)
            np.save(
                os.path.join(self.save_path, "validation_loss.npy"),
                np.array(self.validation_losses),
            )
        self.model.train()

    def training_loop(self) -> None:
        """
        The main training loop script for the autoencoder model.
        """
        start_time = time.time()
        total_iters = 0
        average_loss = 0
        first_kl_epoch = 80
        kl_schedule = [0 for _ in range(first_kl_epoch)] + [
            min(i * 2 / self.n_epochs, 1) for i in range(self.n_epochs - first_kl_epoch)
        ]

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

                for batch_index, batch in progress_bar:
                    try:
                        batch = batch.to(self.device)
                        if batch.dim() == 4:
                            batch = unsqueeze_channels(batch, device=self.device)

                        with autocast(
                            device_type=self._get_device_type(), dtype=torch.float16
                        ):
                            batch = batch.to(self.device)
                            if self.variational:
                                output, kl_loss = self.model(batch, return_kl_loss=True)
                                loss = self.loss_fn(output, batch) + kl_loss * (
                                    1e-7 * kl_schedule[epoch]
                                )  # following the LDM paper

                            else:
                                output = self.model(batch)
                                loss = self.loss_fn(output, batch)

                            scaled_loss = loss / self.gradient_accumulation_steps
                            average_loss += scaled_loss

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

                            self.training_losses.append(average_loss.item())
                            progress_bar.set_postfix(loss=f"{average_loss.item():.4f}")
                            average_loss = 0

                            with torch.no_grad():
                                self._log_train_loss(total_iters)
                                if total_iters % self.save_iters == 0:
                                    self.save_model(total_iters)

                    except Exception as batch_e:
                        self.log(
                            f"Error processing training batch {batch_index} in epoch {epoch}: {batch_e}",
                            level="error",
                            exc_info=True,
                        )
                        continue

                # Compute validation loss and log
                if len(self.val_loader) > 0:
                    self.evaluate(total_iters=total_iters, epoch=epoch)

                self.log(
                    f"Training for epoch {epoch} completed, time spent: {time.time()-epoch_start_time:.2f}s"
                )

            with torch.no_grad():
                self.save_model(total_iters)

            self.log(
                f"Training completed, total time spent: {time.time()-start_time:.2f}s"
            )

        except Exception as e:
            self.log(f"Error in training loop: {e}", level="error", exc_info=True)
            raise

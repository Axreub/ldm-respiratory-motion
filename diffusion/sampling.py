import torch
import numpy as np
from math import prod
from dataclasses import asdict
from args.classes import SamplingArgs
from inference.utils.load_models import load_unet
from diffusion.utils.noise_schedules import get_noise_schedule
from typing import Optional


class DiffusionSampler:
    """
    Class that implements the DDPM and DDIM sampling algorithms.
    """

    def __init__(
        self,
        sampling_args: SamplingArgs,
        backbone_path: str,
        device: str,
    ):
        """
        Initialize the DDPM/DDIM sampler.

        Args:
            sampling_args (SamplingArgs): Configuration arguments for the reverse diffusion sampling process.
            backbone_path (str): The path to the neural network model used for denoising.
            device (str): The device to run the sampling on (e.g., 'cuda' or 'cpu').
        """

        for arg_name, value in asdict(sampling_args).items():
            setattr(self, arg_name, value)
        backbone, model_args, _ = load_unet(backbone_path, device, use_eval=True)

        beta = get_noise_schedule(self.T, model_args.noise_schedule)
        self.alpha = [1 - beta[i] for i in range(self.T)]  # alpha_t = 1-beta_t
        self.ps_alpha = [
            prod(self.alpha[: i + 1]) for i in range(self.T)
        ]  # Product sum alpha list
        self.in_channels = model_args.in_channels
        self.backbone = backbone
        self.device = device

    def generate_sample(
        self,
        input_tensor: torch.Tensor,
        phase_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform  sampling from random noise. If sampling_args.out_channels is less than
        sampling_args.in_channels, the predicted noise in each sampling step will be applied on the last n channels of the sampling tensor,
        where n = sampling_args.out_channels.

        Args:
            input_tensor (torch.Tensor): The noisy tensor at t=T to start sampling from.
            phase_batch (torch.Tensor): The phase batch tensor.
        Returns:
            The sampled tensor corresponding to t=0.

        """

        x_t = input_tensor

        if self.sampling_algo == "ddpm":
            for t in range(self.T, 0, -1):
                x_t = self._ddpm_reverse_step(x_t, t, phase_batch)
            x_0 = x_t

        if self.sampling_algo == "ddim":
            assert (
                type(self.delta_t) is int
            ), f"Error: Sampling algo is set to DDIM, type of delta_t is not int: {type(self.delta_t)}."

            assert (self.delta_t > 0) and (
                self.delta_t <= self.T
            ), f"Error: delta_t is out of range [0, T]: {self.delta_t}"

            for t in range(self.T, 0, -self.delta_t):
                x_t = self._ddim_reverse_step(x_t, t, phase_batch)

            x_0 = x_t

        return x_0

    def _ddpm_reverse_step(
        self,
        x_t: torch.Tensor,
        t: int,
        phase_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single reverse diffusion step for DDPM from timestep t to t-1.

        Args:
            x_t (torch.Tensor): The noisy input tensor at timestep t.
            t (int): The current timestep in the reverse diffusion process (should be in [1, T]).

        Returns:
            torch.Tensor: The denoised tensor at timestep t-1.
        """
        if t > 1:
            z = torch.randn_like(x_t, device=self.device)[:, 1:, :, :, :]
        else:
            z = torch.zeros_like(x_t, device=self.device)[:, 1:, :, :, :]

        pred_noise = self.backbone(
            x_t, torch.tensor([t], device=self.device) / self.T, phase_batch
        )
        ps_alpha_t = self.ps_alpha[t - 1]
        alpha_t = self.alpha[t - 1]

        div_factor = 1 / (np.sqrt(alpha_t) + self.epsilon)
        noise_scaling = (1 - alpha_t) / (np.sqrt(1 - ps_alpha_t) + self.epsilon)
        sigma_t = np.sqrt(1 - alpha_t)

        x_t_minus_1 = self._sum_to_last_channels(
            x_t, div_factor, -div_factor * noise_scaling * pred_noise + sigma_t * z
        )

        return x_t_minus_1

    def _ddim_reverse_step(
        self,
        x_t: torch.Tensor,
        t: int,
        phase_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single reverse diffusion step for DDIM from timestep t to t - delta_t.

        Args:
            x_t (torch.Tensor): The noisy input tensor at timestep t.
            t (int): The current timestep in the reverse diffusion process (should be in [1, T]).

        Returns:
            torch.Tensor: The denoised tensor at timestep t - delta_t.
        """
        t_next = t - self.delta_t

        pred_noise = self.backbone(
            x_t, torch.tensor([t], device=self.device) / self.T, phase_batch
        )
        ps_alpha_t = self.ps_alpha[t - 1]

        ps_alpha_t_next = (
            self.ps_alpha[t_next] if (t_next >= 1) else 1
        )  # ps_alpha_0 is 1 by definition

        x_scale_factor = np.sqrt(ps_alpha_t_next / (ps_alpha_t + self.epsilon))
        noise_factor = np.sqrt(1 - ps_alpha_t_next) - np.sqrt(
            (1 - ps_alpha_t) * ps_alpha_t_next / (ps_alpha_t + self.epsilon)
        )

        x_t_next = self._sum_to_last_channels(
            x_t, x_scale_factor, noise_factor * pred_noise
        )

        return x_t_next

    def _sum_to_last_channels(
        self, x_t: torch.Tensor, x_scale_factor: float, scaled_pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a scaling factor to the DVF (deformation vector field) channels of the input tensor and adds predicted noise to these channels, while leaving the image channel unchanged.

        Args:
            x_t (torch.Tensor): Input tensor of shape (batch, channels, D, H, W), where the first channel is the image and the remaining channels are DVF.
            x_scale_factor (float): Multiplicative scaling factor for the DVF channels.
            scaled_pred_noise (torch.Tensor): Noise tensor to be added to the scaled DVF channels. Should match the shape of the DVF channels.

        Returns:
            torch.Tensor: Output tensor of the same shape as x_t, with the image channel unchanged and the DVF channels scaled and noise-added.
        """

        x_image = x_t[:, :3, :, :, :]
        x_dvf = x_t[:, 3:, :, :, :] * x_scale_factor
        x_dvf += scaled_pred_noise
        sum = torch.cat((x_image, x_dvf), dim=1)
        return sum

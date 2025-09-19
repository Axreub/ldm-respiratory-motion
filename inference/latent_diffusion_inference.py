import torch
import json
import time
import numpy as np
from pathlib import Path

import random
from diffusion.sampling import DiffusionSampler
from inference.utils.load_models import load_unet, load_autoencoder
from inference.utils.load_data import load_samples
from utils.path_obtainer import get_paths
from utils.arg_obtainer import get_args
from visualization.overlays import DVFOverlay
from inference.utils.save_outputs import ensure_empty_dir
import matplotlib.pyplot as plt
from scipy import stats


class LatentDiffusionGenerator:
    def __init__(self):
        latent_diffusion_paths = get_paths("inference")["latent_diffusion"]
        backbone_path = latent_diffusion_paths["diffusion_model_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_autoencoder, _, _ = load_autoencoder(
            latent_diffusion_paths["image_autoencoder_path"],
            device=self.device,
        )
        self.dvf_autoencoder, _, _ = load_autoencoder(
            latent_diffusion_paths["dvf_autoencoder_path"],
            device=self.device,
        )
        self.sampling_args, _, self.current_train_args = get_args("diffusion")
        self.backbone, self.model_args, self.train_args = load_unet(
            backbone_path, self.device
        )
        self.sampler = DiffusionSampler(self.sampling_args, backbone_path, self.device)

    def generate(
        self,
        image: torch.Tensor = None,
        phase: torch.Tensor = None,
        verbose: bool = False,
        generate_overlay=False,
    ) -> torch.Tensor:
        if image == None:
            start_index = random.randint(0, 300)
            latent_image, latent_dvf, phase, _ = load_samples(
                self.train_args,
                input_data_path=None,
                dataloader_type="test",
                n_samples=self.sampling_args.n_samples,
                data_type="latent",
                start_index=start_index,
            )
            image_batch, dvf_batch, phase, paths = load_samples(
                self.train_args,
                input_data_path=Path(
                    str(self.train_args.input_data_path).replace(
                        "latent_idc_downloads", "idc_downloads"
                    )
                ),
                dataloader_type="test",
                n_samples=self.sampling_args.n_samples,
                start_index=start_index,
            )
        else:
            latent_image = self.image_autoencoder.encode(image).unsqueeze(0)

        starting_noise = torch.randn(
            self.sampling_args.n_samples,
            self.model_args.out_channels,
            self.model_args.image_depth,
            self.model_args.image_width,
            self.model_args.image_width,
            device=self.device,
        )
        sampling_input = torch.cat((latent_image, starting_noise), dim=1)

        print("Sampling initialized.")
        start_time = time.time()
        sampling_output = self.sampler.generate_sample(sampling_input, phase)
        inference_time = time.time() - start_time
        print(
            f"Inference complete for {self.sampling_args.n_samples} samples, time taken: {np.round(inference_time, 3)} s"
        )

        diffusion_output = sampling_output[
            :, self.model_args.in_channels - self.model_args.out_channels :, :, :, :
        ]

        if verbose:
            print(
                f"diffusion dvf mean: {diffusion_output.mean()}, ground truth latent dvf mean: {latent_dvf_batch.mean()} "
            )
            print(
                f"diffusion dvf std: {diffusion_output.std()}, ground truth latent dvf std: {latent_dvf_batch.std()}"
            )
            print(
                f"diffusion dvf max: {diffusion_output.max()}, ground truth latent dvf max: {latent_dvf_batch.max()}"
            )
            print(
                f"diffusion dvf min: {diffusion_output.min()}, ground truth latent dvf min: {latent_dvf_batch.min()}"
            )

            ks_stat, p_value = stats.kstest(
                diffusion_output.reshape(-1).cpu().numpy(), "norm", args=(0, 1)
            )
            print(f"KS statistic: {ks_stat}, p-value: {p_value}")

        dvf_output = self.dvf_autoencoder.decode(diffusion_output)[
            : self.sampling_args.n_samples, :, :, :, :
        ]
        dvf_output /= self.model_args.dvf_scale_factor
        if generate_overlay:
            overlay = DVFOverlay(max_arrows=1600, scale=0.01, width=0.003, alpha=0.5)

            ensure_empty_dir("./overlays")

            for sample_idx in range(self.sampling_args.n_samples):
                metadata = json.load(
                    open(paths[sample_idx].with_name("scan_info.json"))
                )

                spacing = (
                    metadata["slice_thickness"],
                    metadata["pixel_spacing"][1],
                    metadata["pixel_spacing"][0],
                )
                if image == None:
                    overlay.render_dvf_overlay(
                        out_dir=f"./overlays/sample_{sample_idx}/decoded_control",
                        vol=image_batch[sample_idx, :, :, :, :].squeeze(dim=0),
                        dvf=self.dvf_autoencoder.decode(latent_dvf)[
                            sample_idx, :, :, :, :
                        ]
                        / self.model_args.dvf_scale_factor,
                        views=("axial", "coronal", "sagittal"),
                        spacing_mm=spacing,
                    )
                    overlay.render_dvf_overlay(
                        out_dir=f"./overlays/sample_{sample_idx}/control",
                        vol=image_batch[sample_idx, :, :, :, :].squeeze(dim=0),
                        dvf=dvf_batch[sample_idx, :, :, :, :],
                        views=("axial", "coronal", "sagittal"),
                        spacing_mm=spacing,
                    )

                overlay.render_dvf_overlay(
                    out_dir=f"./overlays/sample_{sample_idx}/generated_sample",
                    vol=image_batch[sample_idx, :, :, :, :].squeeze(dim=0),
                    dvf=dvf_output[sample_idx, :, :, :, :],
                    views=("axial", "coronal", "sagittal"),
                    spacing_mm=spacing,
                )

        return dvf_output


if __name__ == "__main__":
    inference = LatentDiffusionGenerator()
    samples = inference.generate(generate_overlay=True)

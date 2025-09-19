import torch
import json
import time
import numpy as np

from diffusion.sampling import DiffusionSampler
from inference.utils.load_models import load_unet
from inference.utils.load_data import load_samples
from utils.path_obtainer import get_paths
from utils.arg_obtainer import get_args
from visualization.overlays import DVFOverlay
from inference.utils.save_outputs import ensure_empty_dir
from data_processing.dicom.normalizer import resample_batched_tensor


diffusion_paths = get_paths("inference")["non_latent_diffusion"]
backbone_path = diffusion_paths["model_path"]

device = "cuda" if torch.cuda.is_available() else "cpu"

sampling_args, _, current_train_args = get_args("diffusion")
sampling_args.n_samples = 1

backbone, model_args, train_args = load_unet(backbone_path, device)

image_batch, dvf_batch, phase_batch, paths_list = load_samples(
    train_args,
    input_data_path=None,
    dataloader_type="train",
    n_samples=sampling_args.n_samples,
)  # If custom data path, set input_data_path to current_train_args.input_data_path


orig_image_batch = image_batch.clone()
orig_dvf_batch = dvf_batch.clone()

if train_args.resample_target_dim is not None:
    image_batch = resample_batched_tensor(image_batch, train_args.resample_target_dim)
    dvf_batch = resample_batched_tensor(dvf_batch, train_args.resample_target_dim)


starting_noise = torch.randn_like(dvf_batch)

sampling_input = torch.cat((image_batch, starting_noise), dim=1)

sampler = DiffusionSampler(sampling_args, backbone_path, device)


print("Sampling initialized.")
start_time = time.time()
sampling_output = sampler.generate_sample(sampling_input, phase_batch)
inference_time = time.time() - start_time
print(
    f"Inference complete for {sampling_args.n_samples} samples, time taken: {np.round(inference_time, 3)} s"
)

orig_dvf_batch /= model_args.dvf_scale_factor
sampling_output[:, -3:, :, :, :] /= model_args.dvf_scale_factor


overlay = DVFOverlay(max_arrows=1600, scale=0.01, width=0.003, alpha=0.5)

ensure_empty_dir("./sampling_outputs")
ensure_empty_dir("./overlays")

for sample_idx in range(sampling_args.n_samples):
    metadata = json.load(open(paths_list[sample_idx].with_name("scan_info.json")))

    spacing = (
        metadata["slice_thickness"],
        metadata["pixel_spacing"][1],
        metadata["pixel_spacing"][0],
    )

    sampling_output_paths = overlay.render_dvf_overlay(
        out_dir=f"./overlays/sample_{sample_idx}/generated_sample",
        vol=orig_image_batch[sample_idx, :, :, :, :].squeeze(dim=0),
        dvf=sampling_output[sample_idx, -model_args.out_channels :, :, :, :],
        views=("axial", "coronal", "sagittal"),
        spacing_mm=spacing,
    )

    control_output_paths = overlay.render_dvf_overlay(
        out_dir=f"./overlays/sample_{sample_idx}/control",
        vol=orig_image_batch[sample_idx, :, :, :, :].squeeze(dim=0),
        dvf=orig_dvf_batch[sample_idx, :, :, :, :],
        views=("axial", "coronal", "sagittal"),
        spacing_mm=spacing,
    )

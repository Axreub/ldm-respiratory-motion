import torch
import random
from backbones.blocks.model_parts import SamplingLayer
from inference.utils.load_models import load_autoencoder
from utils.path_obtainer import get_paths
from visualization.overlays import DVFOverlay
from data_processing.data_loaders import MedicalDataLoader
from inference.utils.save_outputs import ensure_empty_dir
import matplotlib.pyplot as plt


dvf_autoencoder_paths = get_paths("inference")["dvf_autoencoder"]
dvf_autoencoder_path = dvf_autoencoder_paths["model_path"]


device = "cuda" if torch.cuda.is_available() else "cpu"

dvf_auto_encoder, encoder_args, train_args = load_autoencoder(
    save_path=dvf_autoencoder_path,
    device=device,
)
train_args.batch_size = 1

train_loader, val_loader, test_loader = MedicalDataLoader(
    train_args=train_args, paired=True, dvf_scale_factor=encoder_args.dvf_scale_factor
).get_dataloaders()

overlay = DVFOverlay(max_arrows=1600, scale=0.01, width=0.003, alpha=0.5)

ensure_empty_dir("./dvf_autoenc_samples")

random_sample_idx = random.randint(0, len(test_loader.dataset))
vol_sample, dvf_sample, _ = test_loader.dataset[random_sample_idx]

vol_sample = vol_sample.to(device)
dvf_sample = dvf_sample.to(device).unsqueeze(0)


dvf_auto_encoder.sampling_layer.identity_sampling = False

pred_latent, kl = dvf_auto_encoder.encode(dvf_sample)

plt.figure()
plt.hist(pred_latent.cpu().detach().numpy().flatten(), bins=100)
plt.title("Histogram of Predicted Latent")
plt.xlabel("Latent Value")
plt.ylabel("Frequency")
plt.savefig(f"./dvf_autoenc_samples/latent_histogram.png")
plt.close()

print(
    f"Encoded dvf std: {torch.std(pred_latent)} encoded dvf mean: {torch.mean(pred_latent)}, encoded dvf max: {torch.max(pred_latent)}, min: {torch.min(pred_latent)}"
)

plt.figure(figsize=(8, 5))
plt.hist(
    pred_latent.clamp(low.item(), high.item()).reshape(-1).cpu(),
    bins=200,
    density=True,
    alpha=0.7,
    edgecolor="black",
)
plt.title("Histogram of values (clamped)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)

# Save to file
plt.savefig("pred_latent_hist.png", dpi=300, bbox_inches="tight")
plt.close()

pred_dvf = dvf_auto_encoder(dvf_sample).squeeze(0) / encoder_args.dvf_scale_factor


metadata = test_loader.dataset.get_metadata(random_sample_idx)

spacing = (
    metadata["slice_thickness"],
    metadata["pixel_spacing"][1],
    metadata["pixel_spacing"][0],
)

sampling_output_paths = overlay.render_dvf_overlay(
    out_dir=f"./dvf_autoenc_samples/sample_{random_sample_idx}/generated_sample",
    vol=vol_sample.squeeze(0),
    dvf=pred_dvf,
    views=("axial", "coronal", "sagittal"),
    spacing_mm=spacing,
)

control_output_paths = overlay.render_dvf_overlay(
    out_dir=f"./dvf_autoenc_samples/sample_{random_sample_idx}/control",
    vol=vol_sample.squeeze(0),
    dvf=dvf_sample.squeeze(0).squeeze(0) / encoder_args.dvf_scale_factor,
    views=("axial", "coronal", "sagittal"),
    spacing_mm=spacing,
)

import torch
from pathlib import Path
from tqdm import tqdm
import shutil
import json

from backbones.models import Autoencoder
from inference.utils.load_models import load_autoencoder
from data_processing.utils.dp_utils import get_file_paths
from data_processing.data_loaders import PairedDataset
from utils.path_obtainer import get_paths


def safe_save_tensor(t: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)


def create_latent_set(
    input_path: Path,
    output_path: Path,
    image_autoencoder: Autoencoder,
    dvf_autoencoder: Autoencoder,
    device: torch.device,
    dvf_scale_factor: float,
    image_autoencoder_path: str,
    dvf_autoencoder_path: str,
):
    """
    Create a latent set from a given input path. Has the same folder structure as the input path, but with latent data.
    """
    paths = get_file_paths([str(input_path)])
    dataset = PairedDataset(paths, dvf_scale_factor=dvf_scale_factor)
    output_paths = [
        Path(str(dataset.input_paths[i]).replace(str(input_path), str(output_path)))
        for i in range(len(dataset))
    ]

    for i in tqdm(range(len(dataset)), desc="Processing latent set"):
        try:
            vol, dvf, _ = dataset[i]
            vol = vol.to(device)
            dvf = dvf.to(device)

            latent_vol = image_autoencoder.encode(vol.unsqueeze(0)).squeeze(0)
            latent_dvf = dvf_autoencoder.encode(
                dvf.unsqueeze(0), return_kl_loss=False
            ).squeeze(0)

            output_vol_path = output_paths[i].with_name("volume.pt")
            output_dvf_path = output_paths[i].with_name("dvf.pt")
            output_mask_path = output_paths[i].with_name("mask.pt")
            output_scan_path = output_paths[i].with_name("scan_info.json")

            mask_path = dataset.input_paths[i].with_name("mask.pt")

            scan_path = dataset.input_paths[i].with_name("scan_info.json")
            scan_info = json.load(open(scan_path))
            scan_info["latent_vol_channels"] = int(latent_vol.shape[0])
            scan_info["latent_dvf_channels"] = int(latent_dvf.shape[0])
            scan_info["latent_vol_spatial_dims"] = list(latent_vol.shape[1:])
            scan_info["latent_dvf_spatial_dims"] = list(latent_dvf.shape[1:])
            scan_info["dvf_scale_factor"] = dvf_scale_factor
            scan_info["image_autoencoder_path"] = image_autoencoder_path
            scan_info["dvf_autoencoder_path"] = dvf_autoencoder_path

            safe_save_tensor(latent_vol.cpu(), output_vol_path)
            safe_save_tensor(latent_dvf.cpu(), output_dvf_path)

            shutil.copy(mask_path, output_mask_path)
            with open(output_scan_path, "w") as f:
                json.dump(scan_info, f)
        except Exception as e:
            print(f"Error processing {dataset.input_paths[i]}: {e}")
            continue


if __name__ == "__main__":
    root = get_paths("data_processing")["idc_downloads"]["processed"]
    output_path = get_paths("data_processing")["idc_downloads"]["latent"]
    latent_diffusion_paths = get_paths("inference")["latent_diffusion"]
    image_autoencoder_path = latent_diffusion_paths["image_autoencoder_path"]
    dvf_autoencoder_path = latent_diffusion_paths["dvf_autoencoder_path"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_autoencoder, _, _ = load_autoencoder(
        save_path=image_autoencoder_path,
        device=device,
    )
    dvf_autoencoder, dvf_encoder_args, _ = load_autoencoder(
        save_path=dvf_autoencoder_path,
        device=device,
    )
    print(f"Creating a latent set with image autoencoder at{image_autoencoder_path}")
    print(f"Creating a latent set with dvf autoencoder at{dvf_autoencoder_path}")

    create_latent_set(
        input_path=root,
        output_path=output_path,
        image_autoencoder=image_autoencoder,
        dvf_autoencoder=dvf_autoencoder,
        device=device,
        dvf_scale_factor=dvf_encoder_args.dvf_scale_factor,
        image_autoencoder_path=image_autoencoder_path,
        dvf_autoencoder_path=dvf_autoencoder_path,
    )

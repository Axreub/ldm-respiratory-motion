import torch
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt


def save_tensor_as_jpg(
    image: torch.Tensor,
    filename: str,
) -> None:
    """
    Save a single-channel image tensor in [-1, 1] as a JPEG file.

    Args:
        image (torch.Tensor): Image tensor in [-1, 1], shape (H, W) or (1, H, W).
        filename (str): Output JPEG file path.

    Returns: None
    """
    # Ensure tensor is on CPU and detached from computation graph before converting to numpy
    image = image.detach().cpu()
    image = torch.clamp(image, -1, 1)
    print(
        f"saved image with mean: {torch.mean(image)} max: {torch.max(image)} min: {torch.min(image)}: std: {torch.std(image)}"
    )

    img = image.squeeze().numpy()
    # Scale from [-1, 1] to [0, 255]
    img = ((img + 1) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(img, mode="L").save(filename, "JPEG")


def ensure_empty_dir(dir_name: str) -> None:
    """Ensure a directory exists and is empty by removing all contained files and subdirectories."""

    if os.path.exists(dir_name):
        for file in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(dir_name)
                os.mkdir(dir_name)
    else:
        os.makedirs(dir_name)

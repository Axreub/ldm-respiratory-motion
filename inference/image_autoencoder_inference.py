import torch
import matplotlib.pyplot as plt
from data_processing.data_loaders import MedicalDataLoader
from inference.utils.load_models import load_autoencoder
from utils.path_obtainer import get_paths


def display_image(
    img: torch.Tensor, pos: int, title: str, num_plots: int, printing: bool = False
) -> None:
    """
    Display a single image in a matplotlib subplot.

    Args:
        img (torch.Tensor): The image tensor to display. Expected shape: (C, H, W).
        pos (int): The position index for the subplot.
        title (str): The title for the subplot.
        num_plots (int): Total number of subplots in the figure.
        printing (bool, optional): If True, print image statistics. Defaults to False.
    """
    plt.subplot(1, num_plots, pos)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np, cmap="gray", vmin=-1, vmax=1.0)  # Normalize range if necessary
    plt.title(title)
    plt.axis("off")

    if printing:
        print(f"{title} Stats:")
        print(f"  Min: {img_np.min():.4f}")
        print(f"  Max: {img_np.max():.4f}")
        print(f"  Mean: {img_np.mean():.4f}")
        print(f"  Std: {img_np.std():.4f}")


if __name__ == "__main__":
    autoenc_type = "image"

    if autoenc_type == "image":
        paths = get_paths("inference")["image_autoencoder"]
    elif autoenc_type == "dvf":
        paths = get_paths("inference")["dvf_autoencoder"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    auto_encoder, encoder_args, train_args = load_autoencoder(
        save_path=paths["model_path"],
        device=device,
    )
    train_loader, val_loader, test_loader = MedicalDataLoader(
        train_args=train_args, paired=False, data_type="image"
    ).get_dataloaders()

    original_img = sample = (
        test_loader.dataset[0].to(torch.float32).to(device).unsqueeze(0)
    )

    clean_img = original_img.clone()
    output = auto_encoder(original_img)

    images = [
        (clean_img.squeeze(0)[:, :, 80, :], "Original Image"),
        (output.squeeze(0)[:, :, 80, :], "Autoencoder Output"),
    ]
    num_plots = len(images)

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(images, start=1):
        display_image(img, i, title, num_plots)
    plt.tight_layout()
    plt.savefig("result1.png")
    plt.show()

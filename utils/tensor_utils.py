import torch


def unsqueeze_channels(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Unsqueezes the channel dimension (4th from last) and moves tensor to the specified device."""
    # Doesn't matter if it's [B, P, C, N_slices, H, W] or [B, C, N_slices, H, W], 4th from last will always be channels
    return x.unsqueeze(-4).to(device)

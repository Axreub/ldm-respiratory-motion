import torch


def get_noise_schedule(T: int, schedule_type: str = "cosine") -> torch.Tensor:
    if schedule_type == "cosine":
        return cosine_beta_schedule(T)
    elif schedule_type == "linear":
        return linear_beta_schedule(T)
    else:
        print(
            f"Error: invalid noise schedule type: {schedule_type}. Possible values: 'cosine', 'linear'"
        )
        raise


def cosine_beta_schedule(T: int) -> torch.Tensor:
    """
    Compute the cosine beta schedule for diffusion models.

    Args:
        T (int): The total number of diffusion steps.

    """
    s: float = 0.008
    steps: torch.Tensor = torch.linspace(0, T, T + 1, dtype=torch.float32)
    alphas_cumprod: torch.Tensor = (
        torch.cos(((steps / T) + s) / (1 + s) * (torch.pi / 2)) ** 2
    )
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)


def linear_beta_schedule(T: int) -> torch.Tensor:
    scale = 1000 / T
    beta_min = scale * 1e-4
    beta_max = scale * 0.02
    return torch.linspace(beta_min, beta_max, T + 1, dtype=torch.float32)

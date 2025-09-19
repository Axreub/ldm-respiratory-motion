import torch
import os
import traceback
import torch.multiprocessing as mp
from training.trainer.diffusion import DiffusionTrainer
from data_processing.data_loaders import MedicalDataLoader
from args.classes import DiffusionArgs, DiffusionTrainArgs
from utils.arg_obtainer import get_args
from backbones.models import UNet
from training.utils.train_utils import make_deterministic, save_args_to_json
from training.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    prepare_dataloader,
    prepare_model,
)
from inference.utils.load_models import load_unet


def train(
    rank: int,
    world_size: int,
    train_args: DiffusionTrainArgs,
    diffusion_args: DiffusionArgs,
    deterministic: bool = False,
) -> None:
    """
    Distributed training loop for diffusion models.

    Args:
        rank (int): The rank of the current process in distributed training.
        world_size (int): Total number of processes participating in training.
        train_args (Any): Training arguments/configuration object.
        diffusion_args (Any): Diffusion model arguments/configuration object.
        deterministic (bool, optional): If True, sets deterministic training for reproducibility. Defaults to False.

    Returns:
        None
    """

    setup_distributed(rank, world_size)
    data_type = "latent" if diffusion_args.image_width != 256 else None
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    if deterministic:
        make_deterministic()

    if train_args.use_warm_checkpoint:
        base_model, _, _ = load_unet(
            train_args.warm_checkpoint_path, device, use_eval=False
        )
    else:
        base_model = UNet(
            model_args=diffusion_args, dropout_rate=train_args.dropout_rate
        )

    compiled_model = torch.compile(base_model)
    model = prepare_model(compiled_model, rank)

    train_loader, val_loader, test_loader = MedicalDataLoader(
        train_args,
        paired=True,
        data_type=data_type,
        dvf_scale_factor=diffusion_args.dvf_scale_factor,
    ).get_dataloaders()

    train_loader = prepare_dataloader(train_loader, rank, world_size)
    val_loader = prepare_dataloader(val_loader, rank, world_size)
    test_loader = prepare_dataloader(test_loader, rank, world_size)

    trainer = DiffusionTrainer(
        model,
        train_args,
        diffusion_args.noise_schedule,
        train_loader,
        val_loader,
        test_loader,
        device=device,
    )

    trainer.training_loop()

    cleanup_distributed()


if __name__ == "__main__":
    _, diffusion_args, train_args = get_args("diffusion")

    if train_args.use_warm_checkpoint:
        _, prev_diffusion_args, prev_train_args = load_unet(
            train_args.warm_checkpoint_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        save_args_to_json(
            prev_diffusion_args,
            prev_train_args,
            "prev_args.json",
            save_dir=train_args.save_path,
        )
        diffusion_args = prev_diffusion_args  # Can't mutate original model architecture

    if train_args.num_data_loader_workers > 0:
        mp.set_start_method("spawn")

    deterministic = False

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1  # Use single process for CPU

    save_args_to_json(diffusion_args, train_args)
    try:
        mp.spawn(
            train,
            args=(world_size, train_args, diffusion_args, deterministic),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        print(f"\n Error in mother process when spawning subprocesses: {e}")
        traceback.print_exc()
        os._exit(1)
    finally:
        cleanup_distributed()

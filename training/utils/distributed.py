import os
import torch
import signal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from training.utils.decorators import rank_zero
from torch.utils.data import DataLoader
from typing import Any, Optional


def setup_distributed(rank: int = 0, world_size: int = 1) -> None:
    """
    Initialize distributed training environment.

    Args:
        rank (int, optional): The rank of the current process. Defaults to 0.
        world_size (int, optional): The total number of processes. Defaults to 1.

    Returns:
        None
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, sigint_handler)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except Exception as e:
        print(f"[Rank {rank}] Error during dist.init_process_group: {e}")
        os._exit(1)

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{rank}")


@rank_zero
def cleanup_distributed() -> None:
    """
    Clean up distributed training by destroying process groups, then killing the current process.

    Returns:
        None
    """
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error while destroying process group: {e}")
    os.kill(os.getpid(), signal.SIGKILL)  # Forcefully kill process


def prepare_dataloader(
    dataloader: DataLoader, rank: int = 0, world_size: int = 1
) -> DataLoader:
    """
    Wrap a DataLoader with a DistributedSampler for distributed training.

    Args:
        dataloader (DataLoader): The original DataLoader to wrap.
        rank (int, optional): The rank of the current process. Defaults to 0.
        world_size (int, optional): The total number of processes. Defaults to 1.

    Returns:
        DataLoader: A new DataLoader using DistributedSampler.
    """
    sampler = DistributedSampler(
        dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    # Create a new DataLoader with the distributed sampler
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        prefetch_factor=dataloader.prefetch_factor,
    )


def prepare_model(model: Any, rank: int = 0) -> Any:
    """
    Wrap a model with DistributedDataParallel (DDP) for distributed training.

    Args:
        model (Any): The model to wrap.
        rank (int, optional): The rank of the current process. Defaults to 0.

    Returns:
        Any: The model wrapped with DDP if CUDA is available, otherwise the original model.
    """
    if torch.cuda.is_available():
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    return model


def sigint_handler(signum: int, frame: Optional[object]) -> None:
    """
    Signal handler for SIGINT (Ctrl+C) during distributed training.

    Args:
        signum (int): The signal number.
        frame (object, optional): The current stack frame.

    Returns:
        None
    """
    print(f"\nRank {dist.get_rank()} received SIGINT (ctrl+c), exiting.")
    cleanup_distributed()

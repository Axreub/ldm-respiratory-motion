import torch
import torch.multiprocessing as mp
import traceback
from backbones.models import Autoencoder, VariationalAutoencoder
from inference.utils.load_models import load_autoencoder
from training.trainer.autoenc import EncoderTrainer
from data_processing.data_loaders import MedicalDataLoader
from utils.arg_obtainer import get_args
from training.utils.train_utils import make_deterministic, save_args_to_json
from training.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    prepare_dataloader,
    prepare_model,
)
import os
from args.classes import EncoderArgs, EncoderTrainArgs


def train(
    rank: int,
    world_size: int,
    train_args: EncoderTrainArgs,
    encoder_args: EncoderArgs,
    deterministic: bool = False,
) -> None:
    """Performs autoencoder training on a single process."""
    setup_distributed(rank, world_size)
    try:
        if deterministic:
            make_deterministic()

        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        if train_args.use_warm_checkpoint:
            base_model, _, _ = load_autoencoder(
                train_args.warm_checkpoint_path, device, use_eval=False
            )

        elif encoder_args.variational:
            base_model = VariationalAutoencoder(
                encoder_args=encoder_args, dropout_rate=train_args.dropout_rate
            )
        else:
            base_model = Autoencoder(
                encoder_args=encoder_args, dropout_rate=train_args.dropout_rate
            )

        prepared_model = prepare_model(base_model, rank)
        model = torch.compile(prepared_model)
        if encoder_args.in_channels == 1:
            data_type = "volume"
        elif encoder_args.in_channels == 3:
            data_type = "dvf"

        train_loader, val_loader, test_loader = MedicalDataLoader(
            train_args,
            paired=False,
            data_type=data_type,
            dvf_scale_factor=encoder_args.dvf_scale_factor,
        ).get_dataloaders()

        train_loader = prepare_dataloader(train_loader, rank, world_size)
        val_loader = prepare_dataloader(val_loader, rank, world_size)
        test_loader = prepare_dataloader(test_loader, rank, world_size)

        trainer = EncoderTrainer(
            model,
            train_args,
            train_loader,
            val_loader,
            test_loader,
            device,
            encoder_args.variational,
        )
        trainer.training_loop()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[Rank {rank}] CUDA OOM error caught. Attempting graceful shutdown.")
        else:
            raise  # Re-raise other exceptions
    except KeyboardInterrupt:
        print(f"[Rank {rank}] Interrupted by user.")
    except Exception as e:
        print(f"[Rank {rank}] ran into an error: {e}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    encoder_args, train_args = get_args("autoencoder")

    if train_args.use_warm_checkpoint:
        _, prev_encoder_args, prev_train_args = load_autoencoder(
            train_args.warm_checkpoint_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        save_args_to_json(
            prev_encoder_args,
            prev_train_args,
            "prev_args.json",
            save_dir=train_args.save_path,
        )

        encoder_args = prev_encoder_args  # Can't mutate original model architecture

    if train_args.num_data_loader_workers > 0:
        mp.set_start_method("spawn")

    deterministic = True

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1  # Use single process for CPU

    save_args_to_json(encoder_args, train_args)
    try:
        mp.spawn(
            train,
            args=(world_size, train_args, encoder_args, deterministic),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        print(f"\n Error in mother process when spawning subprocesses: {e}")
        traceback.print_exc()
        os._exit(1)
    finally:
        cleanup_distributed()

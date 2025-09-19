from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SamplingArgs:
    epsilon: float = field(
        metadata={
            "help": "Epsilon for the sampling algorithm to prevent division by zero"
        }
    )
    delta_t: int = field(
        metadata={"help": "Delta t for the sampling algorithm. Only usable for DDIM"}
    )
    n_samples: int = field(metadata={"help": "Number of samples to generate"})
    T: int = field(
        metadata={
            "help": "Number of timesteps for the forward/backward diffusion process"
        }
    )
    sampling_algo: str = field(
        default="ddim",
        metadata={"help": "Sampling algorithm to use. Possible values: 'ddpm', 'ddim'"},
    )


@dataclass
class DiffusionArgs:
    image_width: int = field(metadata={"help": "Width of the image"})
    image_depth: int = field(metadata={"help": "Depth of the image"})
    in_channels: int = field(
        default=1, metadata={"help": "Number of channels in the input image"}
    )
    n_layers: int = field(
        default=3, metadata={"help": "Number of down- and upsampling layers"}
    )
    base_channels: int = field(
        default=16, metadata={"help": "Number of channels in the base layer"}
    )
    out_channels: int = field(
        default=1, metadata={"help": "Number of channels in the output image"}
    )
    layer_type: bool = field(
        default="simple",
        metadata={
            "help": "What type of convolutional layer to use. Possible values: 'double', 'double_res' and 'simple' (default)"
        },
    )
    keep_z: List[bool] = field(
        default_factory=lambda: [False, True, True, True],
        metadata={"help": "Whether to keep the z-dimension in each layer"},
    )

    noise_schedule: str = field(
        default="cosine",
        metadata={"help": "Noise schedule to use. Possible values: 'cosine'"},
    )
    norm: str = field(
        default="identity",
        metadata={
            "help": "Normalization layer to use. Possible values: 'batch', 'instance', 'group' or None"
        },
    )
    dvf_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Factor by which DFVs are scaled during training, in order to alter the range of values they encompass."
        },
    )


@dataclass
class DiffusionTrainArgs:
    T: int = field(
        metadata={
            "help": "Number of timesteps for the forward/backward diffusion process"
        }
    )
    save_path: str = field(metadata={"help": "Path to save the model"})
    input_data_path: str = field(
        metadata={"help": "Path to the training/val/test data"}
    )

    batch_size: int = field(metadata={"help": "Batch size for the training process"})
    train_ratio: float = field(
        metadata={
            "help": "Ratio of the training data. Train + val + test should sum to 1"
        }
    )
    val_ratio: float = field(
        metadata={
            "help": "Ratio of the validation data. Train + val + test should sum to 1"
        }
    )
    test_ratio: float = field(
        metadata={"help": "Ratio of the test data. Train + val + test should sum to 1"}
    )
    num_data_loader_workers: int = field(
        metadata={
            "help": "Number of workers for the data loader. 0 means no multiprocessing. Recommended to set < number of physical cores"
        }
    )
    n_epochs: int = field(metadata={"help": "Number of training epochs"})
    save_iters: int = field(
        metadata={"help": "Number of iterations between model saves"}
    )
    optimizer: str = field(
        default="adamw",
        metadata={
            "help": "Optimizer to use for the training process. Possible values: 'adamw', 'sgd'"
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={
            "help": "Loss function to use for the training process. Possible values: 'mse'"
        },
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Number of steps for linear warmup"}
    )
    lr_max: float = field(default=1e-4, metadata={"help": "Maximum learning rate"})
    lr_min: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate. Only used for LR scheduling"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Amount of iterations that the training script averages the gradient over before updating it."
        },
    )
    dropout_rate: float = field(
        default=0.0, metadata={"help": "Dropout rate for the training process"}
    )
    loss_fn: str = field(
        default="mse",
        metadata={
            "help": "Loss function to use for the training process. Possible values: 'mse"
        },
    )
    metrics_t: Optional[int] = field(
        default=50, metadata={"help": "Number of timesteps to compute metrics at"}
    )
    data_loader_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used to deterministically reconstruct dataloaders. Useful for train/val/test separation on trained models."
        },
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "Learning rate scheduler to use for the training process. Possible values: 'cosine' or None"
        },
    )
    resample_target_dim: Optional[tuple[int, int, int]] = field(
        default=None,
        metadata={
            "help": "Dimensionality to resample the DVF and volume tensors to during diffusion training. Given in the order [D, H, W]."
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={
            "help": "Loss function to use for the training process. Possible values: 'mse'"
        },
    )
    use_warm_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to use a warm checkpoint for the training process"},
    )
    warm_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the warm checkpoint to use for the training process. If None, no warm checkpoint is used."
        },
    )


@dataclass
class EncoderArgs:
    variational: bool = field(
        default=False,
        metadata={"help": "Whether to use a variational autoencoder"},
    )
    in_channels: int = field(
        default=1, metadata={"help": "Number of channels in the input image"}
    )
    base_channels: int = field(
        default=16, metadata={"help": "Number of channels in the base layer"}
    )
    latent_channels: int = field(
        default=1, metadata={"help": "Number of channels in the latent space"}
    )
    out_channels: int = field(
        default=1, metadata={"help": "Number of channels in the output image"}
    )
    layer_type: bool = field(
        default="simple",
        metadata={
            "help": "What type of convolutional layer to use. Possible values: 'double', 'double_res' and 'simple' (default)"
        },
    )
    keep_z: List[bool] = field(
        default_factory=lambda: [False, True, True],
        metadata={"help": "Whether to keep the z-dimension in each layer"},
    )
    channel_multipliers: int = field(
        default_factory=lambda: [1, 2, 4],
        metadata={"help": "Channel multipliers in the Autoencoder"},
    )
    kernel_size: int = field(
        default=3, metadata={"help": "Kernel size for the convolution layers"}
    )
    stride: int = field(
        default=1, metadata={"help": "Stride for the convolution layers"}
    )
    padding: int = field(
        default=1, metadata={"help": "Padding for the convolution layers"}
    )
    norm: str = field(
        default="identity",
        metadata={
            "help": "Normalization layer to use. Possible values: 'batch', 'instance', 'group'. Defaults to 'identity'."
        },
    )
    dvf_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Factor by which DFVs are scaled during training, in order to alter the range of values they encompass."
        },
    )


@dataclass
class EncoderTrainArgs:
    save_path: str = field(metadata={"help": "Path to save the autoencoder model to"})
    input_data_path: str = field(
        metadata={"help": "Path to the training/val/test data"}
    )
    batch_size: int = field(metadata={"help": "Batch size for the training process"})
    train_ratio: float = field(
        metadata={
            "help": "Ratio of the training data. Train + val + test should sum to 1"
        }
    )
    val_ratio: float = field(
        metadata={
            "help": "Ratio of the validation data. Train + val + test should sum to 1"
        }
    )
    test_ratio: float = field(
        metadata={"help": "Ratio of the test data. Train + val + test should sum to 1"}
    )
    num_data_loader_workers: int = field(
        metadata={
            "help": "Number of workers for the data loader. 0 means no multiprocessing. Recommended to set < number of physical cores"
        }
    )
    n_epochs: int = field(metadata={"help": "Number of training epochs"})
    save_iters: int = field(
        metadata={"help": "Number of iterations between model saves"}
    )
    slice_views: list = field(
        default_factory=lambda: ["axial"],
        metadata={
            "help": "Which plane(s) to use when generating slices for perceptual loss. Choose any combination of ['axial', 'coronal', 'sagittal']"
        },
    )
    optimizer: str = field(
        default="adamw",
        metadata={
            "help": "Optimizer to use for the training process. Possible values: 'adamw', 'sgd'"
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={
            "help": "Loss function to use for the training process. Possible values: 'mse', 'l1', 'lpe', or 'dvf' for DVF autoencoder training"
        },
    )
    l1_factor: float = field(
        default=0.5,
        metadata={"help": "Weight factor for the L1-part of LPE loss."},
    )
    edge_factor: float = field(
        default=0.2,
        metadata={"help": "Weight factor for the edge kernel loss."},
    )
    perceptual_loss_type: str = field(
        default="lpips",
        metadata={
            "help": "Loss type for perceptual loss. Possible values: 'ssim', 'lpips'"
        },
    )

    warmup_steps: int = field(
        default=0, metadata={"help": "Number of steps for linear warmup"}
    )
    lr_max: float = field(default=1e-4, metadata={"help": "Maximum learning rate"})
    lr_min: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate. Only used for LR scheduling"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Amount of iterations that the training script averages the gradient over before updating it."
        },
    )
    dropout_rate: float = field(
        default=0.0, metadata={"help": "Dropout rate for the training process"}
    )

    data_loader_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used to deterministically reconstruct dataloaders. Useful for train/val/test separation on trained models."
        },
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "Learning rate scheduler to use for the training process. Possible values: 'cosine' or None"
        },
    )
    use_warm_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to use a warm checkpoint for the training process"},
    )
    warm_checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "Path to the warm checkpoint to use for the training process. If None, no warm checkpoint is used."
        },
    )

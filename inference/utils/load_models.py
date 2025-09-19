import torch
import json
from typing import Tuple, Union, Optional

from args.classes import (
    DiffusionTrainArgs,
    EncoderTrainArgs,
    DiffusionArgs,
    EncoderArgs,
)
from backbones.models import UNet, Autoencoder, VariationalAutoencoder


def load_unet(
    model_file_path: str,
    device: Union[torch.device, str],
    use_eval: bool = True,
    dropout_rate: Optional[float] = None,
) -> Tuple[UNet, DiffusionArgs, DiffusionTrainArgs]:
    """
    Loads a UNet model from a checkpoint file, along with its associated model and training arguments.
    Assumes that checkpoint file is stored in the same folder as the args.json file.

    Args:
        model_file_path (str): Path to the saved model checkpoint file. (Should end in .pt)
        device (Union[torch.device, str]): The device to load the model onto (e.g., 'cpu', 'cuda', or torch.device).
        use_eval (bool, optional): If True, sets the model to evaluation mode and disables gradients. Defaults to True.
        dropout_rate (Optional[float], optional): If provided, overrides the dropout rate in the loaded model.

    Returns:
        Tuple[UNet, DiffusionArgs, DiffusionTrainArgs]:
            - The loaded UNet model.
            - The model arguments used to construct the UNet (DiffusionArgs).
            - The training arguments used during training (DiffusionTrainArgs).
    """

    model_state = torch.load(model_file_path, weights_only=True)
    model_state = revert_model_weight_names(model_state)
    args_filepath = model_file_path.rsplit("/", 1)[0] + "/args.json"

    with open(args_filepath, "r") as f:
        args_dict = json.load(f)

    model_args = DiffusionArgs(**args_dict["model_args"])
    train_args = DiffusionTrainArgs(**args_dict["train_args"])

    u_net = UNet(model_args=model_args, dropout_rate=dropout_rate).to(device)
    u_net.load_state_dict(model_state)
    compiled_u_net = torch.compile(u_net)
    compiled_u_net = u_net.to(device)

    if use_eval:
        compiled_u_net.eval()
        for param in compiled_u_net.parameters():
            param.requires_grad = False

    return compiled_u_net, model_args, train_args


def load_autoencoder(
    save_path: str,
    device: Union[torch.device, str],
    use_eval: bool = True,
    dropout_rate: Optional[float] = None,
) -> Tuple[Autoencoder, EncoderArgs, EncoderTrainArgs]:
    """
    Loads an Autoencoder model from a checkpoint file, along with its associated model and training arguments.
    Assumes that the checkpoint file is stored in the same directory as the corresponding args.json file.

    Args:
        save_path (str): Path to the saved Autoencoder model checkpoint file (should end with .pt).
        device (Union[torch.device, str]): The device to load the model onto (e.g., 'cpu', 'cuda', or torch.device).
        use_eval (bool, optional): If True, sets the model to evaluation mode and disables gradients. Defaults to True.
        dropout_rate (Optional[float], optional): If provided, overrides the dropout rate in the loaded model.

    Returns:
        Tuple[Autoencoder, EncoderArgs, EncoderTrainArgs]:
            - The loaded Autoencoder model.
            - The model arguments used to construct the Autoencoder (EncoderArgs).
            - The training arguments used during training (EncoderTrainArgs).
    """

    model_state = torch.load(save_path, weights_only=True)
    model_state = revert_model_weight_names(model_state)

    args_filepath = save_path.rsplit("/", 1)[0] + "/args.json"

    with open(args_filepath, "r") as f:
        args_dict = json.load(f)

    model_args = EncoderArgs(**args_dict["model_args"])
    train_args = EncoderTrainArgs(**args_dict["train_args"])

    if model_args.variational:
        autoenc = VariationalAutoencoder(
            encoder_args=model_args, dropout_rate=dropout_rate
        ).to(device)
    else:
        autoenc = Autoencoder(encoder_args=model_args, dropout_rate=dropout_rate).to(
            device
        )
    autoenc.load_state_dict(model_state)
    compiled_autoenc = torch.compile(autoenc)
    compiled_autoenc = compiled_autoenc.to(device)

    if use_eval:
        compiled_autoenc.eval()
        for param in compiled_autoenc.parameters():
            param.requires_grad = False
    return compiled_autoenc, model_args, train_args


def revert_model_weight_names(state_dict: dict) -> dict:
    """
    Reverts model weight key names in a state_dict that may have been altered by
    DistributedDataParallel (DDP) or torch.compile. Specifically, it removes
    leading 'module.' or '_orig_mod.module.' prefixes from parameter names.

    Args:
        state_dict (dict): The state dictionary containing model weights, possibly with
            DDP or torch.compile prefixes in the keys.

    Returns:
        dict: A new state dictionary with the prefixes removed from the keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "").replace("module", "")
        new_key = new_key.lstrip(".")
        new_state_dict[new_key] = v

    return new_state_dict

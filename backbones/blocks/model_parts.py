import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from backbones.blocks.utils import load_activation, load_norm
from typing import Union


def load_conv(
    in_channels: int,
    out_channels: int,
    keep_z: bool = True,
    activation: str = "elu",
    norm: str = "identity",
    kernel_size: int = 3,
    padding: int = 1,
    layer_type: str = "double",
) -> torch.nn.Module:
    """Function that loads in convolutional layers. Supported values are 'double', 'double_res' and 'simple' (default)."""
    layer_classes = {
        "double": DoubleConv,
        "double_res": ResDoubleConv,
        "simple": SimpleConv,
    }

    if layer_type not in layer_classes:
        raise ValueError(
            f"Layer type '{layer_type}' is not supported. Choose from {list(layer_classes.keys())}."
        )

    layer_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "keep_z": keep_z,
        "activation": activation,
        "norm": norm,
        "kernel_size": kernel_size,
        "padding": padding,
    }

    return layer_classes[layer_type](**layer_kwargs)


class SimpleConv(nn.Module):
    """
    A module consisting of one convolutional layer, followed by normalization and and a freely chosen activation function. Optionally, the kernel can be set to only
    convolve in the spatial (x, y) dimensions, preserving the z-dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        keep_z: bool = False,
        activation: str = "tanh",
        norm: str = "identity",
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple] = 1,
        stride: Union[int, tuple] = 1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            keep_z (bool, optional): If True, use kernel size (1, 3, 3) and padding (0, 1, 1)
                to avoid convolving along the z-dimension. Default: False.
            activation (str): The type of activation function that is used at the end of the module. Defaults to "elu".
            kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3.
            padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 1.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
        """
        super().__init__()
        kernel_size = (1, 3, 3) if keep_z else _triple(kernel_size)
        padding = (0, 1, 1) if keep_z else _triple(padding)
        stride = _triple(stride)
        activation = load_activation(activation)
        norm1 = load_norm(norm=norm, channels=out_channels)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            norm1,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class DoubleConv(nn.Module):
    """
    A module consisting of two consecutive 3D convolutional layers, each followed by
    batch normalization and and a freely chosen activation function. Optionally, the kernel can be set to only
    convolve in the spatial (x, y) dimensions, preserving the z-dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        keep_z: bool = False,
        activation: str = "tanh",
        norm: str = "identity",
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple] = 1,
        stride: Union[int, tuple] = 1,
    ):
        super().__init__()
        kernel_size = (1, 3, 3) if keep_z else _triple(kernel_size)
        padding = (0, 1, 1) if keep_z else _triple(padding)
        stride = _triple(stride)
        activation = load_activation(activation)
        norm1 = load_norm(norm=norm, channels=out_channels)
        norm2 = load_norm(norm=norm, channels=out_channels)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            norm1,
            activation,
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            norm2,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class ResDoubleConv(nn.Module):
    """
    A residual block consisting of two consecutive 3D convolutional layers, each followed by
    batch normalization and a freely chosen activation function. Includes a residual connection, which is either
    an identity mapping or a 3D convolution to match the number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        keep_z: bool = False,
        activation: str = "tanh",
        norm: str = "identity",
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple] = 1,
        stride: Union[int, tuple] = 1,
    ):
        super().__init__()
        kernel_size = (1, 3, 3) if keep_z else _triple(kernel_size)
        padding = (0, 1, 1) if keep_z else _triple(padding)
        stride = _triple(stride)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = load_norm(norm=norm, channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = load_norm(norm=norm, channels=out_channels)
        self.residual = (
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = load_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        out = x + x_res
        out = self.activation(out)

        return out


class Downform(nn.Module):
    """Downsampling block for 3D U-Net architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        keep_z: bool = False,
        activation: str = "tanh",
        norm: str = "identity",
        kernel_size: int = 3,
        padding: int = 1,
        layer_type: str = "simple",
    ):
        super().__init__()
        pool_kernel = (1, 2, 2) if keep_z else (2, 2, 2)
        pool_stride = (1, 2, 2) if keep_z else (2, 2, 2)
        self.downform = nn.Sequential(
            nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride),
            load_conv(
                in_channels,
                out_channels,
                keep_z,
                activation,
                norm,
                kernel_size,
                padding,
                layer_type,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downform(x)
        return x


class Upform(nn.Module):
    """Upsampling block for 3D U-Net architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        keep_z: bool = False,
        activation: str = "tanh",
        norm: str = "identity",
        kernel_size: int = 3,
        padding: int = 1,
        layer_type: str = "simple",
        pads_xy: int = 0,
        pads_z: int = 0,
    ):
        super().__init__()
        convt_kernel = (1, 2, 2) if keep_z else (2, 2, 2)
        convt_stride = (1, 2, 2) if keep_z else (2, 2, 2)
        self.pads_xy = pads_xy
        self.pads_z = pads_z
        self.upform = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=convt_kernel,
                stride=convt_stride,
            ),
            load_conv(
                out_channels,
                out_channels,
                keep_z,
                activation,
                norm,
                kernel_size,
                padding,
                layer_type,
            ),
        )

    def forward(self, dec: torch.Tensor) -> torch.Tensor:
        dec_upformed = self.upform(dec)
        dec_upformed = F.pad(
            dec_upformed, [0, self.pads_xy, 0, self.pads_xy, 0, self.pads_z], value=0
        )  # only pad to right, bottom, back
        return dec_upformed


class UpformCat(nn.Module):
    """Upsampling block for 3D U-Net architectures with skip connections between encoder and decoder features."""

    def __init__(
        self,
        in_channels,
        out_channels,
        keep_z=False,
        activation="tanh",
        norm="identity",
        kernel_size=3,
        padding=1,
        layer_type="simple",
        pads_xy: bool = 0,
        pads_z: bool = 0,
    ):
        # in_channels = dec channels, or enc_channels * 2
        super().__init__()
        convt_kernel = (1, 2, 2) if keep_z else (2, 2, 2)
        convt_stride = (1, 2, 2) if keep_z else (2, 2, 2)
        self.pads_xy = pads_xy
        self.pads_z = pads_z
        self.conv = load_conv(
            in_channels,
            out_channels,
            keep_z,
            activation,
            norm,
            kernel_size,
            padding,
            layer_type,
        )
        self.upform = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=convt_kernel, stride=convt_stride
        )

    def forward(self, dec: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        dec_upformed = self.upform(dec)
        dec_upformed = F.pad(
            dec_upformed, [0, self.pads_xy, 0, self.pads_xy, 0, self.pads_z], value=0
        )  # only pad to right, bottom, back
        skip = torch.cat((enc, dec_upformed), dim=1)
        out = self.conv(skip)
        return out


class SamplingLayer(nn.Module):
    """Helper "layer" used in VAEs to conduct sampling and KL-divergence loss in latent space"""

    def __init__(self, identity_sampling: bool = False):
        super().__init__()
        self.identity_sampling = identity_sampling

    def forward(self, mean_logvar: torch.Tensor):
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        if self.identity_sampling:
            z = mean
            kl_loss = torch.zeros(1, device=mean.device)

        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
            kl_loss = (
                0.5
                * torch.sum(  # derived explicitly from assumption that p ~N(0,1)
                    mean.pow(2) + std.pow(2) - 1.0 - logvar, dim=[1, 2, 3, 4]
                )
            )
        return z, kl_loss

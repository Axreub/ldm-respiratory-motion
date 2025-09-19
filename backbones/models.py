import torch.nn as nn
import torch
from typing import Optional, List
from args.classes import (
    EncoderArgs,
    DiffusionArgs,
)
from backbones.blocks.model_parts import (
    Downform,
    UpformCat,
    Upform,
    load_conv,
    SamplingLayer,
)
from backbones.blocks.utils import create_pads


class UNet(nn.Module):
    """U-Net architecture with skip connections using concatenation, supporting time-dependent context embeddings."""

    def __init__(
        self,
        model_args: DiffusionArgs,
        dropout_rate: Optional[float] = None,
    ) -> None:
        """
        Args:
            model_args (DiffusionArgs): Model configuration with required attributes.
            dropout_rate (Optional[float], optional): Dropout rate for regularization. Defaults to 0.0 if not specified.
        """
        super().__init__()
        self.dropout_rate = dropout_rate if (dropout_rate is not None) else 0.0
        self.dropout = nn.Dropout3d(p=self.dropout_rate)

        for key, value in model_args.__dict__.items():
            setattr(self, key, value)
        self.pads_xy, self.pads_z = create_pads(
            model_args.keep_z, img_width=self.image_width, img_depth=self.image_depth
        )
        self.t_embs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.base_channels * 2**i),
                    nn.ReLU(True),
                )
                for i in range(self.n_layers)
            ][::-1]
        )
        self.phase_embs = nn.ModuleList(
            [
                nn.Embedding(9, self.base_channels * 2**i)
                for i in range(self.n_layers)
            ][::-1]
        )

        self.input = nn.Conv3d(
            self.in_channels, out_channels=self.base_channels, kernel_size=1
        )
        self.encs = nn.ModuleList(
            [
                Downform(
                    self.base_channels * 2**i,
                    self.base_channels * 2 ** (i + 1),
                    layer_type=self.layer_type,
                    keep_z=model_args.keep_z[i],
                    norm=self.norm,
                    activation="elu",
                )
                for i in range(self.n_layers - 1)
            ]
        )

        self.bottleneck = Downform(
            self.base_channels * 2 ** (self.n_layers - 1),
            self.base_channels * 2 ** (self.n_layers),
            layer_type=self.layer_type,
            keep_z=model_args.keep_z[-1],
            norm=self.norm,
            activation="elu",
        )

        self.decs = nn.ModuleList(
            [
                UpformCat(
                    self.base_channels * 2 ** (i + 1),
                    self.base_channels * 2**i,
                    layer_type=self.layer_type,
                    keep_z=model_args.keep_z[i],
                    norm=self.norm,
                    activation="elu",
                    pads_xy=self.pads_xy[i],
                    pads_z=self.pads_z[i],
                )
                for i in range(self.n_layers)
            ][::-1]
        )
        self.out = nn.Conv3d(self.base_channels, self.out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        reverse_enc_outputs: List[torch.Tensor] = []
        t = t.view(-1, 1, 1).float()
        t_embs = [
            t_emb(t).squeeze(1).unsqueeze(2).unsqueeze(2).unsqueeze(2)
            for t_emb in self.t_embs
        ]

        phase_embs = [
            emb(phase).view(phase.size(0), -1, 1, 1, 1) for emb in self.phase_embs
        ]

        x = self.input(x)
        reverse_enc_outputs.append(x)

        for enc in self.encs:
            x = enc(x)

            x = self.dropout(x)

            reverse_enc_outputs.append(x)
        enc_outputs = reverse_enc_outputs[::-1]
        bottleneck = self.bottleneck(x)

        bottleneck = self.dropout(bottleneck)
        up = self.decs[0](
            bottleneck, enc_outputs[0]
        )  # no time embedding in first upform

        up = self.dropout(up)

        for n in range(self.n_layers - 1):
            up = self.decs[n + 1](up + t_embs[n] + phase_embs[n], enc_outputs[n + 1])
            up = self.dropout(up)

        out = self.out(up + t_embs[-1] + phase_embs[-1])
        return out


class Encoder(nn.Module):
    """Encoder architecture without skip connections, using Downform blocks."""

    def __init__(
        self, encoder_args: EncoderArgs, dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate if (dropout_rate is not None) else 0.0
        self.dropout = nn.Dropout3d(p=self.dropout_rate)

        for key, value in encoder_args.__dict__.items():
            setattr(self, key, value)
        self.n_layers = len(encoder_args.channel_multipliers)
        self.channel_multipliers = [
            1
        ] + encoder_args.channel_multipliers  # Ensure the first multiplier is 1 for input channels

        self.input = nn.Conv3d(
            self.in_channels, out_channels=self.base_channels, kernel_size=1
        )
        self.encs = nn.ModuleList(
            [
                Downform(
                    self.base_channels * self.channel_multipliers[i],
                    self.base_channels * self.channel_multipliers[i + 1],
                    layer_type=self.layer_type,
                    keep_z=encoder_args.keep_z[i],
                    activation="elu",
                    norm=self.norm,
                )
                for i in range(self.n_layers)
            ]
        )
        self.out = nn.Sequential(
            load_conv(
                self.base_channels * self.channel_multipliers[-1],
                self.base_channels * self.channel_multipliers[-1],
                kernel_size=3,
                layer_type=self.layer_type,
                norm=self.norm,
                activation="elu",
            ),
            nn.Conv3d(
                self.base_channels * self.channel_multipliers[-1],
                self.latent_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for enc in self.encs:
            x = enc(x)
        out = self.out(x)
        return out


class Decoder(nn.Module):
    """Decoder architecture without skip connections, using Upform blocks."""

    def __init__(
        self, encoder_args: EncoderArgs, dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate if (dropout_rate is not None) else 0.0
        self.dropout = nn.Dropout3d(p=self.dropout_rate)
        self.pads_xy, self.pads_z = create_pads(encoder_args.keep_z)

        for key, value in encoder_args.__dict__.items():
            setattr(self, key, value)
        self.n_layers = len(encoder_args.channel_multipliers)
        self.channel_multipliers = [
            1
        ] + encoder_args.channel_multipliers  # Ensure the first multiplier is 1 for input channels

        self.start = nn.Sequential(
            nn.Conv3d(
                self.latent_channels,
                self.base_channels * self.channel_multipliers[-1],
                kernel_size=1,
            ),
            load_conv(
                self.base_channels * self.channel_multipliers[-1],
                self.base_channels * self.channel_multipliers[-1],
                layer_type=self.layer_type,
                norm=self.norm,
                activation="elu",
            ),
        )
        self.decs = nn.ModuleList(
            [
                Upform(
                    self.base_channels * self.channel_multipliers[i + 1],
                    self.base_channels * self.channel_multipliers[i],
                    layer_type=self.layer_type,
                    keep_z=encoder_args.keep_z[i],
                    activation="elu",
                    norm=self.norm,
                    pads_xy=self.pads_xy[i],
                    pads_z=self.pads_z[i],
                )
                for i in range(self.n_layers)
            ][::-1]
        )
        self.out = nn.Conv3d(self.base_channels, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        for i, dec in enumerate(self.decs):
            x = dec(x)
            x = self.dropout(x)
        out = self.out(x)
        return out


class Autoencoder(nn.Module):
    """Autoencoder architecture without skip connections, using Downform and Upform blocks."""

    def __init__(
        self, encoder_args: EncoderArgs, dropout_rate: Optional[float] = None
    ) -> None:
        """
        Args:
            encoder_args (EncoderArgs): Arguments specifying the decoder architecture.
            dropout_rate (Optional[float], optional): Dropout rate for regularization. Defaults to 0.0 if not specified.
        """
        super().__init__()
        self.encoder = Encoder(encoder_args, dropout_rate)
        self.decoder = Decoder(encoder_args, dropout_rate)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_args: EncoderArgs,
        dropout_rate: Optional[float] = None,
        identity_sampling=False,
    ) -> None:
        """
        Args:
            encoder_args (EncoderArgs): Arguments specifying the decoder architecture.
            dropout_rate (Optional[float], optional): Dropout rate for regularization. Defaults to 0.0 if not specified.
        """

        super().__init__()

        self.encoder = Encoder(encoder_args, dropout_rate)
        self.identity_sampling = identity_sampling
        self.sampling_layer = SamplingLayer(self.identity_sampling)
        self.quant_conv = nn.Conv3d(
            encoder_args.latent_channels, encoder_args.latent_channels * 2, 1
        )
        self.post_quant_conv = nn.Conv3d(
            encoder_args.latent_channels, encoder_args.latent_channels, 1
        )
        self.decoder = Decoder(encoder_args, dropout_rate)

    def encode(self, x: torch.Tensor, return_kl_loss: bool = True) -> torch.Tensor:
        assert x.ndim == 5, "Input must have shape (B, C, D, H, W)"
        x = self.encoder(x)
        mu_logvar = self.quant_conv(x)
        posterior, kl_loss = self.sampling_layer(mu_logvar)
        if return_kl_loss:
            return posterior, kl_loss
        return posterior

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5, "Input must have shape (B, C, D, H, W)"
        assert x.ndim == 5, "Input must have shape (B, C, D, H, W)"
        z = self.post_quant_conv(x)
        out = self.decoder(z)
        return out

    def forward(self, x: torch.Tensor, return_kl_loss: bool = True) -> torch.Tensor:
        assert x.ndim == 5, "Input must have shape (B, C, D, H, W)"
        z, kl_loss = self.encode(x)
        out = self.decode(z)
        if return_kl_loss:
            return out, kl_loss
        return out

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TransformerEncoder,
    LSTMEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TransformerCTCModule(TDSConvCTCModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        dropout: float = 0.1,
    ) -> None:
        super(TDSConvCTCModule, self).__init__() 
        
        self.save_hyperparameters()

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TransformerEncoder(
                num_features=num_features,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

class LSTMCTCModule(TDSConvCTCModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int,
        num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super(TDSConvCTCModule, self).__init__() 
        self.save_hyperparameters()

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

        num_features = self.NUM_BANDS * mlp_features[-1]
        
        self.encoder = LSTMEncoder(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            self.encoder,
            nn.Linear(self.encoder.out_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )


class CNNBiLSTMCTCModule(pl.LightningModule):
    """
    CNN + BiLSTM encoder with CTC, matching the repo's TDSConvCTCModule conventions.

    - Expects inputs:  (T, N, bands=2, electrode_channels=16, freq)
    - Produces emissions: (T', N, num_classes) as log-probabilities (log_softmax applied)
    - Uses charset().num_classes and charset().null_class for CTC blank
    - Uses the repo decoder + CharacterErrorRates metrics
    - Skips empty targets when updating metrics (prevents CER blow-ups due to division by target_len)
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        cnn_layers_count: int,
        cnn_channels1: int,
        cnn_channels2: int,
        cnn_kernel_size: int,
        cnn_stride: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Decoder 
        self.decoder = instantiate(decoder)

        # Charset definitions for num classes and CTC blank id
        self.num_classes = charset().num_classes
        self.blank_id = charset().null_class

        # Conv params for output-length calculation
        self.cnn_stride = int(cnn_stride)
        self.cnn_kernel_size = int(cnn_kernel_size)
        print("It's me kernel", self.cnn_kernel_size)
        self.cnn_pad = self.cnn_kernel_size // 2

        self.cnn_layers_count = cnn_layers_count
        in_dim = in_features
        layers = []
        
        # CNN over time: Conv1d expects (N, C_in, T)
        for i in range(self.cnn_layers_count):
            # Use cnn_channels1 for first layer, cnn_channels2 for the rest
            out_dim = cnn_channels1 if i == 0 else cnn_channels2
            
            layers.append(nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=self.cnn_kernel_size,
                stride=self.cnn_stride if i == 0 else 1, # Stride only on first layer
                padding=self.cnn_pad,
            ))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
            
        self.cnn_stack = nn.Sequential(*layers)
        
        # self.conv1 = nn.Conv1d(
        #     in_channels=in_features,
        #     out_channels=cnn_channels1,
        #     kernel_size=self.cnn_kernel_size,
        #     stride=self.cnn_stride,
        #     padding=self.cnn_pad,
        # )
        # self.bn1 = nn.BatchNorm1d(cnn_channels1)

        # self.conv2 = nn.Conv1d(
        #     in_channels=cnn_channels1,
        #     out_channels=cnn_channels2,
        #     kernel_size=self.cnn_kernel_size,
        #     stride=1,
        #     padding=self.cnn_pad,
        # )
        # self.bn2 = nn.BatchNorm1d(cnn_channels2)

        # BiLSTM: expects (T, N, F)
        self.lstm = nn.LSTM(
            input_size=out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classifier to charset classes
        self.classifier = nn.Linear(2 * lstm_hidden, self.num_classes)

        # CTC loss (zero_infinity prevents NaNs for hard-to-align cases)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        # Metrics (same pattern as baseline)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _conv1_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths after conv1 (stride may downsample time).

        Conv1d length formula:
          L_out = floor((L_in + 2*pad - (k-1) - 1) / stride + 1)
        """
        L_in = lengths.to(torch.long)
        k = self.cnn_kernel_size
        pad = self.cnn_pad
        s = self.cnn_stride

        if s == 1:
            return torch.clamp(L_in, min=1)

        L_out = torch.floor_divide((L_in + 2 * pad - (k - 1) - 1), s) + 1
        return torch.clamp(L_out, min=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, bands=2, electrode_channels=16, freq)
        returns emissions: (T', N, num_classes) in log-prob space
        """
        T, N, bands, C, freq = inputs.shape

        # Flatten (bands, channels, freq) into per-timestep features
        # Feature dim F = bands * C * freq
        x = inputs.reshape(T, N, bands * C * freq)  # (T, N, F)

        # Conv1d expects (N, F, T)
        x = x.permute(1, 2, 0)  # (N, F, T)

        # CNN feature extractor
        x = self.cnn_stack(x)
        # x = F.relu(self.bn1(self.conv1(x)))  # (N, cnn_channels1, T')
        # x = F.relu(self.bn2(self.conv2(x)))  # (N, cnn_channels2, T')

        # LSTM expects (T', N, F)
        x = x.permute(2, 0, 1)  # (T', N, cnn_channels2)

        x, _ = self.lstm(x)      # (T', N, 2*lstm_hidden)
        logits = self.classifier(x)     # (T', N, num_classes)
        emissions = F.log_softmax(logits, dim=-1)   # (T', N, num_classes)
        return emissions

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)  # (T', N, num_classes)
        T_prime = emissions.shape[0]

        # Emission lengths:
        # - start from input_lengths (window valid lengths)
        # - apply conv1 stride downsampling
        # - clamp to actual T' produced by forward (robust)
        emission_lengths = self._conv1_out_lengths(input_lengths)
        emission_lengths = torch.clamp(emission_lengths, min=1, max=T_prime)

        loss = self.ctc_loss(
            log_probs=emissions,    # (T', N, C)
            targets=targets.transpose(0, 1),   # (T, N) -> (N, T)
            input_lengths=emission_lengths,   # (N,)
            target_lengths=target_lengths,   # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # (Optional) one-time debug print
        if phase == "val" and not hasattr(self, "_debug_printed"):
            print("DEBUG emissions shape:", emissions.shape)
            print("DEBUG emission_lengths[:3]:", emission_lengths[:3].detach().cpu().tolist())
            print("DEBUG target_lengths[:3]:", target_lengths[:3].detach().cpu().tolist())
            print("DEBUG pred[0]:", predictions[0])
            self._debug_printed = True

        # Update metrics (skip empty targets to prevent CER blow-ups)
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        skipped = 0
        for i in range(N):
            if target_lengths_np[i] == 0:
                skipped += 1
                continue
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/skipped_empty_targets", skipped, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        # Match baseline exactly
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
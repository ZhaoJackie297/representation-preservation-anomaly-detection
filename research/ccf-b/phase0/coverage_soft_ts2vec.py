from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
SOFTCLT_REPO = ROOT / "references" / "softclt_repo" / "softclt_ts2vec"
if str(SOFTCLT_REPO) not in sys.path:
    sys.path.insert(0, str(SOFTCLT_REPO))

from models.soft_losses import hier_CL_soft  # noqa: E402
from models import TSEncoder  # noqa: E402
from utils import (  # noqa: E402
    centerize_vary_length_series,
    take_per_row,
    torch_pad_nan,
)


def transform_soft_batch(
    weights: np.ndarray,
    mode: str | None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply three-channel controls to one batch submatrix.

    ``rownorm`` sets the duplicated per-anchor mass to one while preserving
    off-diagonal quality. ``uniform`` preserves the raw row's off-diagonal
    mass but makes its distribution uniform. ``permuq`` preserves the same
    mass and permutes the off-diagonal weights.
    """

    values = np.asarray(weights, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("soft weights must be square")
    if mode in (None, "raw"):
        return values.astype(np.float32)
    if mode not in {"rownorm", "uniform", "permuq"}:
        raise ValueError(f"unknown soft transform: {mode}")
    if rng is None:
        rng = np.random.default_rng(0)

    diagonal = np.diag(values).copy()
    off_diagonal = values.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if mode == "rownorm":
        for row in range(len(values)):
            mass = 2.0 * float(off_diagonal[row].sum())
            denominator = 1.0 + mass
            values[row, :] = off_diagonal[row] / denominator
            values[row, row] = diagonal[row] / denominator
    elif mode == "uniform":
        for row in range(len(values)):
            off_mass = float(off_diagonal[row].sum())
            values[row, :] = off_mass / max(len(values) - 1, 1)
            values[row, row] = diagonal[row]
    else:
        for row in range(len(values)):
            candidate_indices = [index for index in range(len(values)) if index != row]
            shuffled = off_diagonal[row, candidate_indices].copy()
            rng.shuffle(shuffled)
            values[row, candidate_indices] = shuffled
    return values.astype(np.float32)


class CoverageSoftTS2Vec:
    """SoftTS2Vec that trains from an explicit shared user-batch table."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 64,
        hidden_dims: int = 64,
        depth: int = 5,
        device: str | int = "cpu",
        lr: float = 1e-3,
        batch_size: int = 16,
        lambda_: float = 0.5,
        tau_temp: float = 2.0,
        temporal_unit: int = 0,
        after_iter_callback=None,
        after_epoch_callback=None,
        soft_instance: bool = False,
        soft_temporal: bool = False,
    ) -> None:
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.tau_temp = tau_temp
        self.temporal_unit = temporal_unit
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.soft_instance = soft_instance
        self.soft_temporal = soft_temporal
        self.max_train_length = None
        self._net = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        ).to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.n_epochs = 0
        self.n_iters = 0

    def fit_coverage_table(
        self,
        *,
        train_data: np.ndarray,
        soft_labels: np.ndarray | None,
        batch_table: Iterable[Iterable[int]],
        n_epochs: int = 1,
        verbose: bool = False,
        soft_transform: str | None = None,
        soft_seed: int = 0,
    ) -> list[dict[str, float | int]]:
        if train_data.ndim != 3:
            raise ValueError("train_data must be user x time x feature")
        if self.max_train_length is not None:
            raise ValueError(
                "max_train_length is incompatible with user-indexed batch tables"
            )
        if soft_labels is not None and soft_labels.shape != (
            train_data.shape[0],
            train_data.shape[0],
        ):
            raise ValueError("soft_labels shape does not match train_data")

        data = train_data.astype(np.float32, copy=False)
        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)
        if np.isnan(data).all(axis=2).all(axis=1).any():
            raise ValueError(
                "an entirely empty user sequence would break graph indices"
            )

        batches = [tuple(int(index) for index in batch) for batch in batch_table]
        if not batches:
            raise ValueError("batch_table is empty")
        for batch in batches:
            if len(batch) < 2 or len(set(batch)) != len(batch):
                raise ValueError(f"invalid user batch: {batch}")
            if min(batch) < 0 or max(batch) >= len(data):
                raise ValueError(f"batch index outside user range: {batch}")

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        transform_rng = np.random.default_rng(soft_seed)
        batch_logs: list[dict[str, float | int]] = []
        for _ in range(n_epochs):
            cumulative_loss = 0.0
            for batch_indices in batches:
                index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
                x = torch.from_numpy(data[list(batch_indices)]).to(
                    self.device, non_blocking=True
                )
                if soft_labels is None:
                    soft_labels_batch = None
                else:
                    soft_labels_batch = soft_labels[
                        np.ix_(batch_indices, batch_indices)
                    ]
                    soft_labels_batch = transform_soft_batch(
                        soft_labels_batch,
                        soft_transform,
                        transform_rng,
                    )

                ts_l = x.size(1)
                crop_l = np.random.randint(
                    low=2 ** (self.temporal_unit + 1), high=ts_l + 1
                )
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(
                    low=crop_right, high=ts_l + 1
                )
                crop_offset = np.random.randint(
                    low=-crop_eleft,
                    high=ts_l - crop_eright + 1,
                    size=x.size(0),
                )

                optimizer.zero_grad(set_to_none=True)
                out1 = self._net(
                    take_per_row(
                        x,
                        crop_offset + crop_eleft,
                        crop_right - crop_eleft,
                    )
                )
                out1 = out1[:, -crop_l:]
                out2 = self._net(
                    take_per_row(
                        x,
                        crop_offset + crop_left,
                        crop_eright - crop_left,
                    )
                )
                out2 = out2[:, :crop_l]
                loss = hier_CL_soft(
                    out1,
                    out2,
                    soft_labels_batch,
                    lambda_=self.lambda_,
                    tau_temp=self.tau_temp,
                    temporal_unit=self.temporal_unit,
                    soft_temporal=self.soft_temporal,
                    soft_instance=self.soft_instance,
                )
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                loss_value = float(loss.detach().cpu())
                cumulative_loss += loss_value
                self.n_iters += 1
                if soft_labels_batch is None:
                    active_pair_mass = 0.0
                    realized_pairs = 0
                else:
                    off_diagonal = soft_labels_batch.copy()
                    np.fill_diagonal(off_diagonal, 0.0)
                    active_pair_mass = float(off_diagonal.sum())
                    realized_pairs = int(np.count_nonzero(off_diagonal))
                batch_logs.append(
                    {
                        "iteration": self.n_iters,
                        "loss": loss_value,
                        "batch_size": len(batch_indices),
                        "active_pair_mass": active_pair_mass,
                        "realized_pairs": realized_pairs,
                        "batch_hash_input": int(
                            index_tensor.sum().item()
                        ),
                    }
                )
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss_value)

            cumulative_loss /= len(batches)
            self.n_epochs += 1
            if verbose:
                mean_mass = np.mean(
                    [
                        row["active_pair_mass"]
                        for row in batch_logs[-len(batches) :]
                    ]
                )
                print(
                    f"epoch={self.n_epochs} loss={cumulative_loss:.6f} "
                    f"mean_active_pair_mass={mean_mass:.6f}",
                    flush=True,
                )
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cumulative_loss)
        return batch_logs

    def _eval_with_pooling(
        self,
        x: torch.Tensor,
        mask=None,
        slicing=None,
        encoding_window=None,
    ) -> torch.Tensor:
        output = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == "full_series":
            if slicing is not None:
                output = output[:, slicing]
            output = F.max_pool1d(
                output.transpose(1, 2),
                kernel_size=output.size(1),
            ).transpose(1, 2)
        elif isinstance(encoding_window, int):
            output = F.max_pool1d(
                output.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                output = output[:, :-1]
            if slicing is not None:
                output = output[:, slicing]
        elif encoding_window == "multiscale":
            level = 0
            representations = []
            while (1 << level) + 1 < output.size(1):
                pooled = F.max_pool1d(
                    output.transpose(1, 2),
                    kernel_size=(1 << (level + 1)) + 1,
                    stride=1,
                    padding=1 << level,
                ).transpose(1, 2)
                if slicing is not None:
                    pooled = pooled[:, slicing]
                representations.append(pooled)
                level += 1
            output = torch.cat(representations, dim=-1)
        elif slicing is not None:
            output = output[:, slicing]
        return output.cpu()

    def encode(
        self,
        data: np.ndarray,
        *,
        mask=None,
        encoding_window=None,
        causal: bool = False,
        sliding_length=None,
        sliding_padding: int = 0,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError("data must be instance x time x feature")
        batch_size = batch_size or self.batch_size
        n_samples, ts_l, _ = data.shape
        original_training = self.net.training
        self.net.eval()
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        output_batches = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    representations = []
                    for index in range(0, ts_l, sliding_length):
                        left = index - sliding_padding
                        right = index + sliding_length + (
                            sliding_padding if not causal else 0
                        )
                        sliding = torch_pad_nan(
                            x[:, max(left, 0) : min(right, ts_l)],
                            left=-left if left < 0 else 0,
                            right=right - ts_l if right > ts_l else 0,
                            dim=1,
                        )
                        batch_output = self._eval_with_pooling(
                            sliding,
                            mask,
                            slicing=slice(
                                sliding_padding,
                                sliding_padding + sliding_length,
                            ),
                            encoding_window=encoding_window,
                        )
                        representations.append(batch_output)
                    output = torch.cat(representations, dim=1)
                    if encoding_window == "full_series":
                        output = F.max_pool1d(
                            output.transpose(1, 2).contiguous(),
                            kernel_size=output.size(1),
                        ).squeeze(1)
                else:
                    output = self._eval_with_pooling(
                        x,
                        mask,
                        encoding_window=encoding_window,
                    )
                    if encoding_window == "full_series":
                        output = output.squeeze(1)
                output_batches.append(output)
        self.net.train(original_training)
        return torch.cat(output_batches, dim=0).numpy()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.net.load_state_dict(state_dict)


def write_training_log(
    path: Path,
    *,
    batch_logs: list[dict[str, float | int]],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metadata": metadata, "batches": batch_logs},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

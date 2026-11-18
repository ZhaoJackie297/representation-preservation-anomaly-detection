from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOFTCLT_REPO = ROOT / "references" / "softclt_repo" / "softclt_ts2vec"
if str(SOFTCLT_REPO) not in sys.path:
    sys.path.insert(0, str(SOFTCLT_REPO))

from models.soft_losses import inst_CL_soft  # noqa: E402
from models.timelags import dup_matrix  # noqa: E402


def make_soft(batch_size: int, *, diagonal: float = 1.0, off_diagonal: float = 0.5) -> np.ndarray:
    values = np.full((batch_size, batch_size), off_diagonal, dtype=np.float32)
    np.fill_diagonal(values, diagonal)
    return values


def weight_statistics(soft: np.ndarray) -> dict[str, float]:
    values = np.asarray(soft, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("soft weights must be square")
    diagonal = float(np.diag(values).sum())
    total = float(values.sum())
    off_diagonal = total - diagonal
    row_sums = values.sum(axis=1)
    entropy = []
    kl_to_uniform = []
    for user, row in enumerate(values):
        row_off = np.delete(np.asarray(row, dtype=np.float64), user)
        positive = row_off[row_off > 0]
        if positive.size:
            p = positive / positive.sum()
            entropy.append(float(-(p * np.log(p)).sum()))
            uniform = np.full_like(p, 1.0 / p.size)
            kl_to_uniform.append(float((p * np.log(p / uniform)).sum()))
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_max": float(row_sums.max()),
        "diagonal_mass": diagonal,
        "off_diagonal_mass": off_diagonal,
        "off_diagonal_fraction": off_diagonal / total if total else 0.0,
        "off_diagonal_entropy_mean": float(np.mean(entropy)) if entropy else 0.0,
        "off_diagonal_kl_to_uniform_mean": (
            float(np.mean(kl_to_uniform)) if kl_to_uniform else 0.0
        ),
    }


def normalized_rows(soft: np.ndarray) -> np.ndarray:
    values = np.asarray(soft, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each soft-weight row must have positive mass")
    return (values / totals).astype(np.float32)


def toy_losses(soft: np.ndarray, *, seed: int = 7) -> dict[str, float]:
    torch.manual_seed(seed)
    z1 = torch.randn(soft.shape[0], 3, 4, dtype=torch.float64)
    z2 = torch.randn(soft.shape[0], 3, 4, dtype=torch.float64)
    hard_weights = np.eye(soft.shape[0], dtype=np.float32)
    def loss_for(values: np.ndarray) -> float:
        label_l, label_r = dup_matrix(
            torch.as_tensor(values, dtype=torch.float64)
        )
        return float(inst_CL_soft(z1, z2, label_l, label_r))

    return {
        "loss_raw": loss_for(soft),
        "loss_row_normalized": loss_for(normalized_rows(soft)),
        "loss_identity": loss_for(hard_weights),
        "loss_scaled_2x": loss_for(soft * 2.0),
    }


def batch_audit(
    soft: np.ndarray,
    batch_table: list[list[int]],
) -> dict[str, object]:
    if not batch_table:
        raise ValueError("batch table must not be empty")
    stats_rows = []
    for batch in batch_table:
        if len(batch) < 2 or len(set(batch)) != len(batch):
            raise ValueError(f"invalid batch: {batch}")
        submatrix = soft[np.ix_(batch, batch)]
        stats_rows.append(weight_statistics(submatrix))
    first = stats_rows[0]
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in stats_rows])),
            "min": float(np.min([row[key] for row in stats_rows])),
            "max": float(np.max([row[key] for row in stats_rows])),
        }
        for key in first
    }
    return {
        "batch_count": len(batch_table),
        "batch_size": len(batch_table[0]),
        "aggregate": aggregate,
        "first_batch_toy_losses": toy_losses(
            soft[np.ix_(batch_table[0], batch_table[0])]
        ),
    }


def matrix_audit(
    soft: np.ndarray,
    batch_table: list[list[int]] | None = None,
) -> dict[str, object]:
    stats = weight_statistics(soft)
    losses = toy_losses(soft)
    result: dict[str, object] = {
        "shape": list(soft.shape),
        "weights": stats,
        "toy_losses": losses,
        "row_normalization_changes_loss": not math.isclose(
            losses["loss_raw"],
            losses["loss_row_normalized"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "scaling_is_linear": math.isclose(
            losses["loss_scaled_2x"],
            2.0 * losses["loss_raw"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
    }
    if batch_table is not None:
        result["batch_audit"] = batch_audit(soft, batch_table)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-labels", type=Path)
    parser.add_argument("--batch-table", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.soft_labels is not None:
        soft = np.load(args.soft_labels)
    else:
        soft = make_soft(16)
    batch_table = None
    if args.batch_table is not None:
        batch_table = json.loads(
            args.batch_table.read_text(encoding="utf-8")
        )
    result = matrix_audit(soft, batch_table=batch_table)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

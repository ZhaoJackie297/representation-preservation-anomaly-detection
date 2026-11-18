from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def top_m_entropy(
    prior: np.ndarray,
    top_m: int,
) -> np.ndarray:
    values = np.asarray(prior, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("profile prior must be a square matrix")
    if top_m <= 0 or top_m >= values.shape[0]:
        raise ValueError("top_m must be between 1 and n_users-1")
    indices = np.argsort(-values, axis=1)[:, :top_m]
    support = np.take_along_axis(values, indices, axis=1)
    if np.any(support < 0):
        raise ValueError("profile prior contains negative probabilities")
    totals = support.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("profile prior has an empty support row")
    support = support / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(support > 0, support * np.log(support), 0.0)
    return -terms.sum(axis=1)


def profile_prior_information(
    prior: np.ndarray,
    *,
    top_m: int = 10,
    flat_ratio: float = 0.95,
    majority_fraction: float = 0.80,
) -> dict[str, object]:
    entropy = top_m_entropy(prior, top_m)
    threshold = flat_ratio * np.log(top_m)
    fraction_above = float(np.mean(entropy > threshold))
    mean_entropy = float(np.mean(entropy))
    triggered = bool(
        mean_entropy > threshold
        or fraction_above >= majority_fraction
    )
    return {
        "user_count": int(prior.shape[0]),
        "top_m": top_m,
        "flat_ratio": flat_ratio,
        "flat_threshold": float(threshold),
        "mean_Hm": mean_entropy,
        "median_Hm": float(np.median(entropy)),
        "fraction_users_above_threshold": fraction_above,
        "majority_fraction": majority_fraction,
        "triggered": triggered,
        "status": (
            "FLAT_PROFILE_PRIOR"
            if triggered
            else "PRIOR_INFORMATIVE"
        ),
        "mechanism_interpretation": (
            "UNTESTABLE_NOT_A3_FAILURE"
            if triggered
            else "TESTABLE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-m", type=int, default=10)
    args = parser.parse_args()

    result = {
        "prior_path": str(args.prior),
        "prior_sha256": file_sha256(args.prior),
        **profile_prior_information(
            np.load(args.prior),
            top_m=args.top_m,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

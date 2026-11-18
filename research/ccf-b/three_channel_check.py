from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def official_loss(logits: np.ndarray, weights: np.ndarray) -> float:
    return float(-(weights * np.log(softmax(logits))).sum())


def numerical_gradient(
    function,
    values: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    gradient = np.zeros_like(values)
    for index in range(len(values)):
        left = values.copy()
        right = values.copy()
        left[index] += eps
        right[index] -= eps
        gradient[index] = (function(left) - function(right)) / (2.0 * eps)
    return gradient


def kl_divergence(q: np.ndarray, uniform: np.ndarray) -> float:
    return float((q * np.log(q / uniform)).sum())


def run_algebra_check(
    *,
    trials: int = 2000,
    seed: int = 1,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    errors = {"identity": 0.0, "gradient": 0.0, "diff": 0.0}
    pinsker_violations = 0
    for _ in range(trials):
        negatives = int(rng.integers(3, 40))
        size = negatives + 1
        logits = rng.normal(size=size)
        off_diagonal_weights = rng.random(negatives) * rng.choice(
            [0.05, 0.5, 1.0]
        )
        weights = np.concatenate([[1.0], off_diagonal_weights])
        mass = off_diagonal_weights.sum()
        total = 1.0 + mass
        pi = mass / total
        q = off_diagonal_weights / mass
        target = np.concatenate([[1.0 - pi], pi * q])
        probability = softmax(logits)

        errors["identity"] = max(
            errors["identity"],
            abs(
                official_loss(logits, weights)
                - total * (-(target * np.log(probability)).sum())
            ),
        )
        gradient = total * probability - weights
        errors["gradient"] = max(
            errors["gradient"],
            np.abs(
                numerical_gradient(
                    lambda values: official_loss(values, weights),
                    logits,
                )
                - gradient
            ).max(),
        )
        q_extended = np.concatenate([[0.0], q])
        hard_weights = np.zeros(size)
        hard_weights[0] = 1.0
        hard_gradient = probability - hard_weights
        errors["diff"] = max(
            errors["diff"],
            np.abs(
                (gradient - hard_gradient)
                - mass * (probability - q_extended)
            ).max(),
        )
        uniform = np.ones(negatives) / negatives
        kl = kl_divergence(q, uniform)
        if abs(q - uniform).sum() > np.sqrt(2.0 * kl) + 1e-12:
            pinsker_violations += 1
    return {
        **errors,
        "pinsker_violations": float(pinsker_violations),
        "trials": float(trials),
    }


def run_free_logit_check(
    *,
    negatives: int = 8,
    seed: int = 1,
) -> float:
    rng = np.random.default_rng(seed)
    off_diagonal_weights = rng.random(negatives) * 0.5 + 0.05
    weights = np.concatenate([[1.0], off_diagonal_weights])
    logits = np.zeros(negatives + 1)
    total = weights.sum()
    for _ in range(200_000):
        logits -= 0.05 * (total * softmax(logits) - weights)
    centered_logits = logits[1:] - logits[1:].mean()
    centered_log_weights = np.log(off_diagonal_weights)
    centered_log_weights -= centered_log_weights.mean()
    return float(np.abs(centered_logits - centered_log_weights).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()
    result = {
        "algebra": run_algebra_check(trials=args.trials),
        "free_logit_max_abs_error": run_free_logit_check(),
        "candidate_count_for_batch": "2B-2 negative off-diagonal entries; one diagonal positive entry",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

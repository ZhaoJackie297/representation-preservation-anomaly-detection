from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

CCF_B = Path(__file__).resolve().parents[1]
PHASE0 = Path(__file__).resolve().parent
for path in (str(CCF_B), str(PHASE0)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cert_pilot import load_incident_ids  # noqa: E402
from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402
from dual_end_entry import encode_dataset, evaluate_scores, robust_scores  # noqa: E402


SOFT_VARIANTS = {
    "A1-Cov",
    "A1-Cov-rownorm",
    "A1-Cov-uniform",
    "A1-Cov-permq",
    "A1-Cov-A5a",
}


def load_model(model_dir: Path, *, input_dims: int, soft_instance: bool) -> CoverageSoftTS2Vec:
    model = CoverageSoftTS2Vec(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device="cpu",
        batch_size=16,
        soft_instance=soft_instance,
        soft_temporal=False,
    )
    model.load(str(model_dir / "model.pth"))
    return model


def evaluate_one(
    *,
    model_dir: Path,
    variant: str,
    seed: int,
    values: np.ndarray,
    active_mask: np.ndarray,
    event_id: np.ndarray,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
) -> dict[str, object]:
    model = load_model(
        model_dir,
        input_dims=values.shape[-1],
        soft_instance=variant in SOFT_VARIANTS,
    )
    masked, plain = encode_dataset(model, values, device="cpu")
    raw_error = np.abs(masked - plain).sum(axis=-1).astype(np.float32)
    train_mask = active_mask[:, : train_end + 1] & np.isfinite(
        raw_error[:, : train_end + 1]
    )
    train_errors = raw_error[:, : train_end + 1][train_mask]
    if train_errors.size == 0:
        raise RuntimeError("no active training errors for calibration")
    scores = robust_scores(train_errors, raw_error)
    result: dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "model": str(model_dir / "model.pth"),
        "score_summary": {
            "mean": float(scores.mean()),
            "max": float(scores.max()),
        },
    }
    for split, start, end in (
        ("validation", val_start, val_end),
        ("test", test_start, test_end),
    ):
        result[split] = {
            str(k): evaluate_scores(
                scores=scores,
                event_id=event_id,
                active_mask=active_mask,
                eval_start=start,
                eval_end=end,
                k=k,
            )
            for k in (5, 10, 20)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-data", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--a3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "A0-matched",
            "A0-A3batch",
            "A1-Cov",
            "A1-Cov-rownorm",
            "A1-Cov-uniform",
            "A1-Cov-permq",
        ],
    )
    args = parser.parse_args()

    r42 = args.cert_data / "processed_r42"
    dataset = np.load(r42 / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_start = int(dataset["val_start_index"])
    val_end = int(dataset["val_end_index"])
    test_start = int(dataset["test_start_index"])
    test_end = int(dataset["test_end_index"])
    event_id = load_incident_ids(
        r42 / "r4.2_incidents.csv",
        users,
        dates,
    )

    payload: dict[str, object] = {
        "dataset": str(r42 / "r4.2_sequence_dataset.npz"),
        "evaluation_protocol": {
            "score": "mask_last vs plain L1, training-period robust calibration",
            "k_grid": [5, 10, 20],
            "cooldown_days": 7,
            "splits": ["validation", "test"],
            "test_role": "r4.2 development test, not external confirmation",
        },
        "runs": {},
    }
    for seed in args.seeds:
        seed_results = {}
        for variant in args.variants:
            root = args.a3_root if variant == "A0-A3batch" else args.main_root
            model_dir = root / f"seed_{seed}" / variant
            if not (model_dir / "model.pth").is_file():
                raise FileNotFoundError(model_dir / "model.pth")
            seed_results[variant] = evaluate_one(
                model_dir=model_dir,
                variant=variant,
                seed=seed,
                values=values,
                active_mask=active_mask,
                event_id=event_id,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
            print(f"evaluated seed={seed} variant={variant}", flush=True)
        payload["runs"][str(seed)] = seed_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

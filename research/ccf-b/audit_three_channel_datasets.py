from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CCF_B = Path(__file__).resolve().parent
PHASE0 = CCF_B / "phase0"
if str(PHASE0) not in sys.path:
    sys.path.insert(0, str(PHASE0))
if str(CCF_B) not in sys.path:
    sys.path.insert(0, str(CCF_B))

from audit_cert_profile_information import build_profile_features  # noqa: E402
from audit_softclt_instance_weights import three_channel_statistics  # noqa: E402
from common_coverage_sampler import build_common_coverage_table  # noqa: E402
from dual_end_entry import profile_soft_labels, random_batch_table  # noqa: E402


def aggregate_batches(
    soft: np.ndarray,
    batch_table: tuple[tuple[int, ...], ...] | list[list[int]],
    *,
    source: str,
) -> dict[str, object]:
    rows = [
        three_channel_statistics(soft[np.ix_(batch, batch)])
        for batch in batch_table
    ]
    keys = [key for key in rows[0] if key != "candidate_count"]
    return {
        "source": source,
        "batch_count": len(rows),
        "batch_size": len(batch_table[0]),
        "candidate_count": int(rows[0]["candidate_count"]),
        "aggregate": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "min": float(np.min([row[key] for row in rows])),
                "max": float(np.max([row[key] for row in rows])),
            }
            for key in keys
        },
    }


def random_audit(
    soft: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    batch_count: int,
) -> dict[str, object]:
    batches = random_batch_table(
        user_count=soft.shape[0],
        batch_size=batch_size,
        batches=batch_count,
        seed=seed,
    )
    return aggregate_batches(soft, batches, source="random_shuffle")


def cert_audit(
    *,
    dataset: Path,
    feature_csv: Path,
    seed: int,
    batch_size: int,
    batch_count: int,
) -> dict[str, object]:
    profile, _, _, train_end = build_profile_features(
        dataset_path=dataset,
        feature_csv=feature_csv,
    )
    soft, metadata = profile_soft_labels(profile)
    return {
        "dataset": str(dataset),
        "profile_dimensions": int(profile.shape[1]),
        "train_end_index": int(train_end),
        "soft_sha256": hashlib.sha256(
            soft.astype(np.float32).tobytes()
        ).hexdigest().upper(),
        "soft_metadata": metadata,
        "batch_audit": random_audit(
            soft,
            seed=seed,
            batch_size=batch_size,
            batch_count=batch_count,
        ),
    }


def clue_audit(
    *,
    data_dir: Path,
    seed: int,
    batch_size: int,
    base_steps: int,
) -> dict[str, object]:
    profile = np.load(data_dir / "clue_profile_features.npy")
    soft, metadata = profile_soft_labels(profile)
    with np.load(data_dir / "clue_prior_graphs.npz") as graph:
        targets = {
            name: {
                tuple(sorted((int(edge[0]), int(edge[1]))))
                for edge in graph[name]
            }
            for name in ("A3", "A5a", "A5b")
        }
    coverage = build_common_coverage_table(
        user_count=soft.shape[0],
        target_edge_sets=targets,
        seed=seed,
        stage=0,
        batch_size=batch_size,
        target_coverage=0.8,
        min_training_batches=base_steps,
        max_coverage_batches=10_000,
    )
    return {
        "dataset": str(data_dir / "clue_user_day_dataset.npz"),
        "profile_dimensions": int(profile.shape[1]),
        "soft_sha256": hashlib.sha256(
            soft.astype(np.float32).tobytes()
        ).hexdigest().upper(),
        "soft_metadata": metadata,
        "coverage": coverage.to_summary(),
        "batch_audit": aggregate_batches(
            soft,
            coverage.training_batches,
            source="A3_A5a_A5b_common_coverage",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-data", type=Path, required=True)
    parser.add_argument("--clue-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-batches", type=int, default=100)
    parser.add_argument("--clue-base-steps", type=int, default=530)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "r4.2": cert_audit(
            dataset=args.cert_data / "processed_r42"
            / "r4.2_sequence_dataset.npz",
            feature_csv=args.cert_data / "processed_r42"
            / "r4.2_user_day_features.csv",
            seed=args.seed,
            batch_size=args.batch_size,
            batch_count=args.random_batches,
        ),
        "r5.2": cert_audit(
            dataset=args.cert_data / "processed_r52"
            / "r5.2_sequence_dataset.npz",
            feature_csv=args.cert_data / "processed_r52"
            / "r5.2_user_day_features.csv",
            seed=args.seed + 1,
            batch_size=args.batch_size,
            batch_count=args.random_batches,
        ),
        "CLUE-LDS": clue_audit(
            data_dir=args.clue_data,
            seed=args.seed,
            batch_size=args.batch_size,
            base_steps=args.clue_base_steps,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

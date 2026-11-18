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

from audit_cert_profile_information import build_profile_features  # noqa: E402
from build_prior_graphs import (  # noqa: E402
    degree_preserving_random,
    permuted_prior,
    top_m_edges,
)
from build_profile_prior import gaussian_prior  # noqa: E402
from common_coverage_sampler import build_common_coverage_table  # noqa: E402
from dual_end_entry import (  # noqa: E402
    VARIANTS,
    encode_dataset,
    evaluate_scores,
    profile_soft_labels,
    robust_scores,
    train_variant,
)
from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402


def load_incident_ids(
    incidents_path: Path,
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    event_id = np.full((len(users), len(dates)), -1, dtype=np.int16)
    user_index = {str(user): index for index, user in enumerate(users)}
    date_index = {str(day): index for index, day in enumerate(dates)}
    with incidents_path.open("r", encoding="utf-8", newline="") as handle:
        import csv

        for event, row in enumerate(csv.DictReader(handle)):
            user = user_index.get(row["insider_id"])
            if user is None:
                continue
            start = date_index.get(row["first_observed_date"])
            end = date_index.get(row["last_observed_date"])
            if start is None or end is None:
                continue
            event_id[user, start : end + 1] = event
    return event_id


def graph_targets(profile: np.ndarray, seed: int, top_m: int) -> dict[str, set[tuple[int, int]]]:
    prior, _ = gaussian_prior(profile)
    a3 = top_m_edges(prior, top_m)
    a5a = degree_preserving_random(a3, len(prior), seed + 2)
    permuted, _ = permuted_prior(prior, seed + 1)
    a5b = top_m_edges(permuted, top_m)
    return {"A3": a3, "A5a": a5a, "A5b": a5b}


def build_seed_coverage(
    *,
    profile: np.ndarray,
    seed: int,
    batch_size: int,
    base_steps: int,
    top_m: int,
) -> tuple[object, dict[str, set[tuple[int, int]]]]:
    targets = graph_targets(profile, seed, top_m)
    coverage = build_common_coverage_table(
        user_count=len(profile),
        target_edge_sets=targets,
        seed=seed,
        stage=0,
        batch_size=batch_size,
        target_coverage=0.8,
        min_training_batches=base_steps,
        max_coverage_batches=10_000,
    )
    return coverage, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=[
            "A0-matched",
            "A0-A3batch",
            "A1-Cov",
            "A1-Cov-rownorm",
            "A1-Cov-uniform",
            "A1-Cov-permq",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-steps", type=int, default=620)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--user-limit", type=int)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--device", default="cpu")
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
    if args.user_limit is not None:
        limit = args.user_limit
        values = values[:limit]
        users = users[:limit]
        active_mask = active_mask[:limit]

    profile, _, profile_users, profile_train_end = build_profile_features(
        dataset_path=r42 / "r4.2_sequence_dataset.npz",
        feature_csv=r42 / "r4.2_user_day_features.csv",
    )
    expected_profile_users = profile_users[: len(users)].tolist()
    if expected_profile_users != users.tolist():
        raise ValueError("CERT profile user order differs from sequence data")
    if profile_train_end != train_end:
        raise ValueError("CERT profile train end differs from sequence data")
    profile = profile[: len(users)]
    soft_labels, soft_metadata = profile_soft_labels(profile)
    event_id = load_incident_ids(
        r42 / "r4.2_incidents.csv",
        users,
        dates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "dataset": str(r42 / "r4.2_sequence_dataset.npz"),
        "user_limit": args.user_limit,
        "base_steps": args.base_steps,
        "seeds": args.seeds,
        "soft_metadata": soft_metadata,
        "runs": {},
    }

    for seed in args.seeds:
        full_coverage, targets = build_seed_coverage(
            profile=profile,
            seed=seed,
            batch_size=args.batch_size,
            base_steps=args.base_steps,
            top_m=args.top_m,
        )
        a3_coverage = build_common_coverage_table(
            user_count=len(profile),
            target_edge_sets={"A3": targets["A3"]},
            seed=seed,
            stage=0,
            batch_size=args.batch_size,
            target_coverage=0.8,
            min_training_batches=len(full_coverage.training_batches),
            max_coverage_batches=10_000,
        )
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_summary: dict[str, object] = {
            "coverage": full_coverage.to_summary(),
            "a3_coverage": a3_coverage.to_summary(),
            "models": {},
        }
        for variant in args.variants:
            if variant == "A0-official":
                from dual_end_entry import random_batch_table

                table = random_batch_table(
                    user_count=len(users),
                    batch_size=args.batch_size,
                    batches=args.base_steps,
                    seed=seed,
                )
            elif variant == "A0-A3batch":
                table = a3_coverage.training_batches
            else:
                table = full_coverage.training_batches
            variant_dir = seed_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "batch_table.json").write_text(
                json.dumps(
                    [list(batch) for batch in table],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            model_summary = train_variant(
                variant=variant,
                seed=seed,
                train_values=values[:, : train_end + 1],
                soft_labels=soft_labels,
                batch_table=table,
                output_dir=variant_dir,
                device=args.device,
                batch_size=args.batch_size,
                tau_temp=1.5,
            )
            run_summary["models"][variant] = model_summary
            if args.encode:
                model = CoverageSoftTS2Vec(
                    input_dims=values.shape[-1],
                    output_dims=64,
                    hidden_dims=64,
                    depth=5,
                    device=args.device,
                    batch_size=args.batch_size,
                    soft_instance=variant
                    in (
                        "A1-R0",
                        "A1-Cov",
                        "A1-Cov-A5a",
                        "A1-Cov-uniform",
                        "A1-Cov-rownorm",
                        "A1-Cov-permq",
                    ),
                    soft_temporal=False,
                )
                model.load(str(variant_dir / "model.pth"))
                masked, plain = encode_dataset(model, values, device=args.device)
                raw_error = np.abs(masked - plain).sum(axis=-1).astype(np.float32)
                train_mask = active_mask[:, : train_end + 1] & np.isfinite(
                    raw_error[:, : train_end + 1]
                )
                train_errors = raw_error[:, : train_end + 1][train_mask]
                scores = robust_scores(train_errors, raw_error)
                np.savez_compressed(
                    variant_dir / "scores.npz",
                    users=users,
                    dates=dates,
                    raw_error=raw_error,
                    anomaly_score=scores.astype(np.float32),
                    event_id=event_id,
                    active_mask=active_mask,
                )
                model_summary["validation"] = evaluate_scores(
                    scores=scores,
                    event_id=event_id,
                    active_mask=active_mask,
                    eval_start=val_start,
                    eval_end=val_end,
                )
                model_summary["test"] = evaluate_scores(
                    scores=scores,
                    event_id=event_id,
                    active_mask=active_mask,
                    eval_start=test_start,
                    eval_end=test_end,
                )
            (variant_dir / "summary.json").write_text(
                json.dumps(model_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary["runs"][str(seed)] = run_summary
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

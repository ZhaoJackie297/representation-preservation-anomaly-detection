from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r52"
OUT_DIR = PROCESSED / "tuning_v1"
DATASET = PROCESSED / "r5.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r5.2_user_day_features.csv"
LABELS = PROCESSED / "r5.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"
BASE_RESULTS = PROCESSED / "r5.2_external_results.json"
RESULTS_FILE = OUT_DIR / "r5.2_tuned_sensitivity_results.json"
SUMMARY_FILE = OUT_DIR / "r5.2_tuned_sensitivity_summary.json"

TS_SEEDS = [20261109, 20261110]
IF_SEEDS = list(range(10))
PRIMARY_K = 10
REPORT_K = [5, 10, 20]
COOLDOWN_DAYS = 7

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import (  # noqa: E402
    FEATURES,
    LOG1P_FEATURES,
    robust_calibrate,
)
from deep_compare_per_user_r42 import (  # noqa: E402
    per_user_quantile_scores,
)
from external_eval_r52 import (  # noqa: E402
    build_external_alerts,
    evaluate_external,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402


def train_ts(seed: int) -> None:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"r5.2_ts_depth3_{seed}_raw.npz"
    if path.exists():
        print(f"Already exists: {path}", flush=True)
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    with np.load(DATASET, allow_pickle=False) as dataset:
        values = dataset["values"].astype(np.float32)
        train_end = int(dataset["train_end_index"])
    model = TS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=3,
        device=0 if torch.cuda.is_available() else "cpu",
        batch_size=16,
        lr=0.001,
    )
    loss_log = model.fit(
        values[:, : train_end + 1, :],
        n_epochs=10,
        verbose=False,
    )
    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    plain = np.asarray(
        model.encode(values, **encode_args),
        dtype=np.float32,
    )
    raw_error = np.abs(masked - plain).sum(axis=-1).astype(
        np.float32
    )
    np.savez_compressed(
        path,
        raw_error=raw_error,
        loss_first=np.asarray(float(loss_log[0])),
        loss_last=np.asarray(float(loss_log[-1])),
    )
    print(f"Finished: {path}", flush=True)


def train_iforest() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    values = (
        raw.set_index(["user_id", "date"])[FEATURES]
        .reindex(index)
        .to_numpy(dtype=np.float64)
        .reshape(len(users), len(dates), len(FEATURES))
    )
    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            finite = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][finite] = np.log1p(
                values[:, :, feature_index][finite]
            )

    train_active = active_mask[:, : train_end + 1]
    train_features = values[:, : train_end + 1][train_active]
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std
    train_normalized = normalized[:, : train_end + 1][train_active]

    for seed in IF_SEEDS:
        path = OUT_DIR / f"r5.2_if_100_auto_f100_{seed}_raw.npz"
        if path.exists():
            print(f"Already exists: {path}", flush=True)
            continue
        model = IsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_normalized)
        raw_score = np.full(
            active_mask.shape, np.nan, dtype=np.float32
        )
        raw_score[active_mask] = -model.score_samples(
            normalized[active_mask]
        )
        np.savez_compressed(path, raw_score=raw_score)
        print(f"Finished: {path}", flush=True)


def score_variants(
    raw_score: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    train_active = active_mask[:, : train_end + 1]
    train_values = raw_score[:, : train_end + 1][train_active]
    global_raw = robust_calibrate(train_values, raw_score)
    personal_raw = per_user_quantile_scores(
        raw_score, active_mask, train_end
    )
    return {
        "global_raw": global_raw.astype(np.float32),
        "global_dynamic": compute_risk(
            users, dates, global_raw, active_mask
        ),
        "personal_quantile_raw": personal_raw.astype(np.float32),
        "personal_quantile_dynamic": compute_risk(
            users, dates, personal_raw, active_mask
        ),
    }


def analyze() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        start_index = train_end + 1

    labels_frame = pd.read_csv(LABELS, parse_dates=["date"])
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    labels = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in labels_frame.itertuples(index=False):
        labels[
            user_index[row.user_id],
            date_index[row.date.strftime("%Y-%m-%d")],
        ] = 1
    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "first_observed_date",
            "last_observed_date",
        ],
    )

    models = {}
    ts_scores = []
    for seed in TS_SEEDS:
        path = OUT_DIR / f"r5.2_ts_depth3_{seed}_raw.npz"
        with np.load(path, allow_pickle=False) as saved:
            ts_scores.append(saved["raw_error"].astype(np.float32))
    models["ts2vec_depth3"] = np.nanmean(
        np.stack(ts_scores), axis=0
    ).astype(np.float32)

    if_scores = []
    for seed in IF_SEEDS:
        path = OUT_DIR / f"r5.2_if_100_auto_f100_{seed}_raw.npz"
        with np.load(path, allow_pickle=False) as saved:
            if_scores.append(saved["raw_score"].astype(np.float32))
    models["iforest_100_auto"] = np.nanmean(
        np.stack(if_scores), axis=0
    ).astype(np.float32)

    results = []
    for model_name, raw_score in models.items():
        variants = score_variants(
            raw_score,
            active_mask,
            train_end,
            users,
            dates,
        )
        for variant, score in variants.items():
            for k in REPORT_K:
                alerts = build_external_alerts(
                    score,
                    active_mask,
                    start_index,
                    k,
                    COOLDOWN_DAYS,
                )
                result = evaluate_external(
                    users,
                    dates,
                    active_mask,
                    alerts,
                    labels,
                    incidents,
                    start_index,
                    f"{model_name}_{variant}",
                    k,
                )
                result["model"] = model_name
                result["variant"] = variant
                results.append(result)

    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected = {
        "ts2vec": "ts2vec_depth3",
        "isolation_forest": "iforest_100_auto",
        "variant": "personal_quantile_dynamic",
        "selected_on": "r4.2 validation only",
    }
    summary = {
        "purpose": (
            "Post-hoc external sensitivity for the configurations selected "
            "on r4.2 validation. r5.2 has already been used for evaluation "
            "and is not a fresh confirmatory holdout."
        ),
        "selected_configurations": selected,
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "report_k": REPORT_K,
        "ts2vec_seeds": TS_SEEDS,
        "iforest_seeds": IF_SEEDS,
        "results_file": str(RESULTS_FILE),
        "base_result_file": str(BASE_RESULTS),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["train-ts", "train-iforest", "analyze"],
        required=True,
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.stage == "train-ts":
        if args.seed is None:
            raise ValueError("--seed is required for train-ts")
        train_ts(args.seed)
    elif args.stage == "train-iforest":
        train_iforest()
    else:
        analyze()


if __name__ == "__main__":
    main()

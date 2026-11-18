from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
OUT_DIR = PROCESSED / "tuning_v1"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r4.2_user_day_features.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

CONFIG_SUMMARY_FILE = OUT_DIR / "r4.2_tuning_config_summary.csv"
SEED_METRICS_FILE = OUT_DIR / "r4.2_tuning_seed_metrics.csv"
SELECTION_FILE = OUT_DIR / "r4.2_tuning_selection.csv"
RESULTS_FILE = OUT_DIR / "r4.2_tuning_results.json"
SUMMARY_FILE = OUT_DIR / "r4.2_tuning_summary.json"

TS2VEC_SEEDS = [20261109, 20261110]
IFOREST_SEEDS = list(range(10))

TS2VEC_CONFIGS = [
    {
        "name": "ts_base",
        "output_dims": 64,
        "hidden_dims": 64,
        "depth": 5,
        "epochs": 10,
    },
    {
        "name": "ts_output32",
        "output_dims": 32,
        "hidden_dims": 64,
        "depth": 5,
        "epochs": 10,
    },
    {
        "name": "ts_hidden32_output32",
        "output_dims": 32,
        "hidden_dims": 32,
        "depth": 5,
        "epochs": 10,
    },
    {
        "name": "ts_depth3",
        "output_dims": 64,
        "hidden_dims": 64,
        "depth": 3,
        "epochs": 10,
    },
    {
        "name": "ts_depth7",
        "output_dims": 64,
        "hidden_dims": 64,
        "depth": 7,
        "epochs": 10,
    },
    {
        "name": "ts_epochs20",
        "output_dims": 64,
        "hidden_dims": 64,
        "depth": 5,
        "epochs": 20,
    },
]

IFOREST_CONFIGS = [
    {
        "name": "if_100_auto_f100",
        "n_estimators": 100,
        "max_samples": "auto",
        "max_features": 1.0,
    },
    {
        "name": "if_200_auto_f100",
        "n_estimators": 200,
        "max_samples": "auto",
        "max_features": 1.0,
    },
    {
        "name": "if_400_auto_f100",
        "n_estimators": 400,
        "max_samples": "auto",
        "max_features": 1.0,
    },
    {
        "name": "if_200_sub50_f100",
        "n_estimators": 200,
        "max_samples": 0.5,
        "max_features": 1.0,
    },
    {
        "name": "if_400_sub50_f100",
        "n_estimators": 400,
        "max_samples": 0.5,
        "max_features": 1.0,
    },
    {
        "name": "if_200_auto_f70",
        "n_estimators": 200,
        "max_samples": "auto",
        "max_features": 0.7,
    },
]

FEATURES = [
    "logon_count",
    "logoff_count",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
    "after_hours_logon_count",
    "other_pc_logon_count",
    "usb_connect_count",
    "usb_disconnect_count",
    "file_copy_count",
    "archive_copy_count",
    "executable_copy_count",
    "unique_file_extensions",
    "email_count",
    "email_recipient_count",
    "external_recipient_count",
    "email_attachment_count",
    "email_size_bytes",
    "after_hours_email_count",
    "http_count",
    "after_hours_http_count",
    "external_http_count",
    "unique_http_domains",
]
LOG1P_FEATURES = FEATURES[:2] + FEATURES[5:]

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import (  # noqa: E402
    robust_calibrate,
)
from deep_compare_per_user_r42 import (  # noqa: E402
    per_user_quantile_scores,
)
from production_protocol_r42 import (  # noqa: E402
    COOLDOWN_DAYS,
    HORIZONS,
    PRIMARY_K,
    REPORT_K,
    build_alerts,
    evaluate,
    label_matrix,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402


def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def load_iforest_values(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
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
    return values, raw


def ts2vec_raw_error(
    values: np.ndarray,
    train_end: int,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    import torch

    path = OUT_DIR / f"{config['name']}_{seed}_raw.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return (
                saved["raw_error"].astype(np.float32),
                {
                    "loss_first": float(saved["loss_first"]),
                    "loss_last": float(saved["loss_last"]),
                    "reused": True,
                },
            )

    seed_everything(seed)
    train_values = values[:, : train_end + 1, :]
    device = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=values.shape[-1],
        output_dims=config["output_dims"],
        hidden_dims=config["hidden_dims"],
        depth=config["depth"],
        device=device,
        batch_size=16,
        lr=0.001,
    )
    loss_log = model.fit(
        train_values,
        n_epochs=config["epochs"],
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_error, {
        "loss_first": float(loss_log[0]),
        "loss_last": float(loss_log[-1]),
        "reused": False,
    }


def iforest_raw_score(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    config: dict,
    seed: int,
) -> np.ndarray:
    path = OUT_DIR / f"{config['name']}_{seed}_raw.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["raw_score"].astype(np.float32)

    train_active = active_mask[:, : train_end + 1]
    train_features = values[:, : train_end + 1][train_active]
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std
    train_normalized = normalized[:, : train_end + 1][train_active]
    model = IsolationForest(
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        contamination="auto",
        max_features=config["max_features"],
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
    return raw_score


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


def validation_metrics(
    score: np.ndarray,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    train_end: int,
    val_end: int,
    name: str,
) -> dict:
    alerts = build_alerts(
        score, active_mask, PRIMARY_K, COOLDOWN_DAYS
    )
    result = evaluate(
        users,
        dates,
        active_mask,
        alerts,
        labels,
        incidents,
        train_end,
        val_end,
        name,
        PRIMARY_K,
    )
    validation = result["splits"]["validation"]
    return {
        "recall_3d": validation["horizons"]["3"]["event_recall"],
        "recall_1d": validation["horizons"]["1"]["event_recall"],
        "precision": validation["precision_at_k"],
        "false_per_day": validation["false_alerts_per_day"],
    }


def selection_key(row: dict) -> tuple:
    return (
        -row["validation_recall_3d"],
        -row["validation_recall_1d"],
        -row["validation_precision"],
        row["validation_false_per_day"],
        row["seed_recall_3d_std"],
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        values = dataset["values"].astype(np.float32)
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        val_end = int(dataset["val_end_index"])

    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    seed_rows = []
    config_rows = []
    selected_scores = {}

    for model_name, configs, seeds in (
        ("ts2vec", TS2VEC_CONFIGS, TS2VEC_SEEDS),
        ("isolation_forest", IFOREST_CONFIGS, IFOREST_SEEDS),
    ):
        iforest_values = None
        if model_name == "isolation_forest":
            iforest_values, _ = load_iforest_values(users, dates)

        for config in configs:
            raw_scores = []
            for seed in seeds:
                print(
                    f"[{model_name}] {config['name']} seed={seed}",
                    flush=True,
                )
                if model_name == "ts2vec":
                    raw_score, training = ts2vec_raw_error(
                        values,
                        train_end,
                        config,
                        seed,
                    )
                else:
                    raw_score = iforest_raw_score(
                        iforest_values,
                        active_mask,
                        train_end,
                        config,
                        seed,
                    )
                    training = {}
                raw_scores.append(raw_score)
                variants = score_variants(
                    raw_score,
                    active_mask,
                    train_end,
                    users,
                    dates,
                )
                for variant, score in variants.items():
                    metrics = validation_metrics(
                        score,
                        users,
                        dates,
                        active_mask,
                        labels,
                        incidents,
                        train_end,
                        val_end,
                        f"{config['name']}_{variant}_{seed}",
                    )
                    seed_rows.append(
                        {
                            "model": model_name,
                            "config": config["name"],
                            "seed": seed,
                            "variant": variant,
                            **{
                                f"validation_{key}": value
                                for key, value in metrics.items()
                            },
                            **training,
                        }
                    )

            ensemble_raw = np.nanmean(
                np.stack(raw_scores, axis=0), axis=0
            ).astype(np.float32)
            ensemble_variants = score_variants(
                ensemble_raw,
                active_mask,
                train_end,
                users,
                dates,
            )
            for variant, score in ensemble_variants.items():
                ensemble_metrics = validation_metrics(
                    score,
                    users,
                    dates,
                    active_mask,
                    labels,
                    incidents,
                    train_end,
                    val_end,
                    f"{config['name']}_{variant}_ensemble",
                )
                seed_subset = pd.DataFrame(seed_rows)
                seed_subset = seed_subset[
                    (seed_subset["model"] == model_name)
                    & (seed_subset["config"] == config["name"])
                    & (seed_subset["variant"] == variant)
                ]
                config_rows.append(
                    {
                        "model": model_name,
                        "config": config["name"],
                        "variant": variant,
                        "validation_recall_3d": ensemble_metrics[
                            "recall_3d"
                        ],
                        "validation_recall_1d": ensemble_metrics[
                            "recall_1d"
                        ],
                        "validation_precision": ensemble_metrics[
                            "precision"
                        ],
                        "validation_false_per_day": ensemble_metrics[
                            "false_per_day"
                        ],
                        "seed_recall_3d_mean": float(
                            seed_subset[
                                "validation_recall_3d"
                            ].mean()
                        ),
                        "seed_recall_3d_std": float(
                            seed_subset[
                                "validation_recall_3d"
                            ].std(ddof=0)
                        ),
                        "seed_recall_1d_mean": float(
                            seed_subset[
                                "validation_recall_1d"
                            ].mean()
                        ),
                        "ensemble_raw_score": ensemble_raw,
                    }
                )

        model_rows = sorted(
            [
                row
                for row in config_rows
                if row["model"] == model_name
            ],
            key=selection_key,
        )
        selected = model_rows[0]
        selected_scores[model_name] = {
            "model": model_name,
            "config": selected["config"],
            "variant": selected["variant"],
            "score": ensemble_variants_for_selection(
                selected,
                users,
                dates,
                active_mask,
                train_end,
                values,
                iforest_values,
            ),
            "validation": {
                key: selected[key]
                for key in (
                    "validation_recall_3d",
                    "validation_recall_1d",
                    "validation_precision",
                    "validation_false_per_day",
                    "seed_recall_3d_mean",
                    "seed_recall_3d_std",
                )
            },
        }

    config_frame = pd.DataFrame(config_rows)
    config_frame.drop(
        columns=["ensemble_raw_score"], errors="ignore"
    ).to_csv(CONFIG_SUMMARY_FILE, index=False, encoding="utf-8")
    pd.DataFrame(seed_rows).to_csv(
        SEED_METRICS_FILE, index=False, encoding="utf-8"
    )

    selection_rows = []
    final_results = []
    for model_name, selected in selected_scores.items():
        score = selected.pop("score")
        selected["selection_file"] = str(SELECTION_FILE)
        selection_rows.append(
            {
                "model": model_name,
                "config": selected["config"],
                "variant": selected["variant"],
                **selected["validation"],
            }
        )
        for k in REPORT_K:
            alerts = build_alerts(
                score, active_mask, k, COOLDOWN_DAYS
            )
            result = evaluate(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                f"{selected['config']}_{selected['variant']}",
                k,
            )
            result["model"] = model_name
            result["config"] = selected["config"]
            result["variant"] = selected["variant"]
            final_results.append(result)

    pd.DataFrame(selection_rows).to_csv(
        SELECTION_FILE, index=False, encoding="utf-8"
    )
    RESULTS_FILE.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "purpose": (
            "Controlled r4.2 tuning; validation selects configuration "
            "and calibration, development test is reported only for "
            "the selected configurations."
        ),
        "ts2vec_seeds": TS2VEC_SEEDS,
        "iforest_seeds": IFOREST_SEEDS,
        "ts2vec_configs": TS2VEC_CONFIGS,
        "iforest_configs": IFOREST_CONFIGS,
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "horizons": HORIZONS,
        "selected": selection_rows,
        "selection_file": str(SELECTION_FILE),
        "results_file": str(RESULTS_FILE),
        "config_summary_file": str(CONFIG_SUMMARY_FILE),
        "seed_metrics_file": str(SEED_METRICS_FILE),
        "test_used_for_selection": False,
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def ensemble_variants_for_selection(
    selected: dict,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    train_end: int,
    values: np.ndarray,
    iforest_values: np.ndarray | None,
) -> np.ndarray:
    model_name = selected["model"]
    config_name = selected["config"]
    if model_name == "ts2vec":
        config = next(
            item
            for item in TS2VEC_CONFIGS
            if item["name"] == config_name
        )
        raw_scores = [
            ts2vec_raw_error(
                values, train_end, config, seed
            )[0]
            for seed in TS2VEC_SEEDS
        ]
    else:
        config = next(
            item
            for item in IFOREST_CONFIGS
            if item["name"] == config_name
        )
        raw_scores = [
            iforest_raw_score(
                iforest_values,
                active_mask,
                train_end,
                config,
                seed,
            )
            for seed in IFOREST_SEEDS
        ]
    ensemble_raw = np.nanmean(
        np.stack(raw_scores, axis=0), axis=0
    ).astype(np.float32)
    return score_variants(
        ensemble_raw,
        active_mask,
        train_end,
        users,
        dates,
    )[selected["variant"]]


if __name__ == "__main__":
    main()

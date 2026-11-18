from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
WINDOW_DAYS = 7
DECAY = 0.65
HIGH_SCORE_THRESHOLD = 3.0
SEVERITY_WEIGHT = 1.0
MEMORY_WEIGHT = 0.0
FREQUENCY_WEIGHT = 0.0
PRIMARY_K = 10

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42_cached,
    load_r52_cached,
)


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "masked": OUT / "e17_r42_M0_masked_quantile.npy",
        "channel": "r42_original_channel_top2_quantile",
        "manifold_prefix": "r42",
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
    },
    "r42_retrained": {
        "family": "r42",
        "masked": OUT / "e19_r42_retrained_masked_quantile.npy",
        "channel": "r42_retrained_channel_top2_quantile",
        "manifold_prefix": "r42_retrained",
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
    },
    "r52_external": {
        "family": "r52",
        "masked": OUT / "e17_r52_M0_masked_quantile.npy",
        "channel": "r52_external_channel_top2_quantile",
        "manifold_prefix": "r52",
        "sequence": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
    },
}


def compute_risk_with_components(
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    severity = scores.astype(np.float32)
    memory = np.full(scores.shape, np.nan, dtype=np.float32)
    frequency = np.full(scores.shape, np.nan, dtype=np.float32)
    risk = np.full(scores.shape, np.nan, dtype=np.float32)

    for user_index in range(scores.shape[0]):
        state = 0.0
        history: deque[float] = deque(maxlen=WINDOW_DAYS)
        for time_index in range(scores.shape[1]):
            if not active_mask[user_index, time_index]:
                state = 0.0
                history.clear()
                continue
            value = float(scores[user_index, time_index])
            state = max(value, DECAY * state)
            history.append(value)
            freq = (
                sum(item > HIGH_SCORE_THRESHOLD for item in history)
                / WINDOW_DAYS
            )
            memory[user_index, time_index] = state
            frequency[user_index, time_index] = freq
            risk[user_index, time_index] = (
                SEVERITY_WEIGHT * value
                + MEMORY_WEIGHT * state
                + FREQUENCY_WEIGHT * freq
            )
    return {
        "severity": severity,
        "memory": memory,
        "frequency": frequency,
        "risk": risk,
    }


def threshold_level(
    value: float,
    thresholds: dict[str, float],
) -> int:
    return int(
        sum(
            value >= threshold
            for threshold in thresholds.values()
        )
    )


def queue_band(rank: int) -> str:
    if rank <= 3:
        return "critical"
    if rank <= 7:
        return "elevated"
    return "watch"


def main() -> None:
    queue = pd.read_csv(
        OUT / "v2m_s_daily_queue_k10.csv",
        dtype={"user_id": str},
    )
    channel_store = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    families = {
        "r42": load_r42_cached(),
        "r52": load_r52_cached(),
    }

    evidence_frames = []
    threshold_rows = []
    component_summary = {}
    saved_arrays = {}

    for block_name, config in BLOCKS.items():
        data = families[config["family"]]
        active_mask = data["active_mask"]
        train_end = (
            int(data["train_end"])
            if "train_end" in data
            else int(data["test_start"]) - 1
        )
        masked_q = np.load(config["masked"]).astype(np.float32)
        manifold_q = np.mean(
            np.stack(
                [
                    np.load(
                        OUT
                        / (
                            f"e19_{config['manifold_prefix']}_"
                            f"learned_manifold_quantile_{seed}.npy"
                        )
                    )
                    for seed in SEEDS
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        channel_q = channel_store[config["channel"]].astype(
            np.float32
        )
        triple_q = (
            masked_q + manifold_q + channel_q
        ) / 3.0
        components = compute_risk_with_components(
            triple_q, active_mask
        )
        percentile = daily_percentile(
            components["risk"], active_mask
        ).astype(np.float32)

        # The reconstructed risk must exactly match the risk used by V2-M.
        # This assertion is an implementation check, not an experimental
        # parameter.
        official_percentile = np.load(
            OUT / "v2m_s_components.npz",
            allow_pickle=False,
        )[f"{block_name}_signal"].astype(np.float32)
        reconstruction_error = float(
            np.nanmax(
                np.abs(
                    percentile[active_mask]
                    - official_percentile[active_mask]
                )
            )
        )
        if reconstruction_error > 1e-6:
            raise RuntimeError(
                f"{block_name}: risk reconstruction error "
                f"{reconstruction_error}"
            )

        train_values = components["risk"][
            :, : train_end + 1
        ][active_mask[:, : train_end + 1]]
        thresholds = {
            "level_1_95pct": float(np.quantile(train_values, 0.95)),
            "level_2_98pct": float(np.quantile(train_values, 0.98)),
            "level_3_99pct": float(np.quantile(train_values, 0.99)),
        }

        block_queue = queue[
            queue["data_block"] == block_name
        ].copy()
        user_index = {
            str(user): index
            for index, user in enumerate(data["users"])
        }
        date_index = {
            date.strftime("%Y-%m-%d"): index
            for index, date in enumerate(data["dates"])
        }
        rows = []
        for row in block_queue.itertuples(index=False):
            user = user_index[str(row.user_id)]
            time = date_index[str(row.date)]
            risk_value = float(components["risk"][user, time])
            rows.append(
                {
                    "data_block": block_name,
                    "split": row.split,
                    "date": row.date,
                    "rank": int(row.rank),
                    "user_id": str(row.user_id),
                    "source": row.source,
                    "ts_score": float(row.ts_score),
                    "iforest_score": float(row.iforest_score),
                    "triple_quantile": float(triple_q[user, time]),
                    "severity": float(components["severity"][user, time]),
                    "memory": float(components["memory"][user, time]),
                    "frequency": float(
                        components["frequency"][user, time]
                    ),
                    "dynamic_risk": risk_value,
                    "risk_percentile": float(
                        percentile[user, time]
                    ),
                    "threshold_level": threshold_level(
                        risk_value, thresholds
                    ),
                    "queue_band": queue_band(int(row.rank)),
                }
            )
        evidence_frames.append(pd.DataFrame(rows))

        split_ranges = (
            {
                "external_test": (
                    data["test_start"],
                    len(data["dates"]) - 1,
                )
            }
            if block_name == "r52_external"
            else {
                "validation": (
                    data["train_end"] + 1,
                    data["val_end"],
                ),
                "test": (
                    data["val_end"] + 1,
                    len(data["dates"]) - 1,
                ),
            }
        )
        for split, (start, end) in split_ranges.items():
            split_active = active_mask[:, start : end + 1]
            split_risk = components["risk"][:, start : end + 1]
            for level, threshold in thresholds.items():
                flagged = split_active & (split_risk >= threshold)
                threshold_rows.append(
                    {
                        "data_block": block_name,
                        "split": split,
                        "level": level,
                        "threshold": threshold,
                        "alerted_account_days": int(flagged.sum()),
                        "alerts_per_day": float(
                            flagged.sum() / flagged.shape[1]
                        ),
                    }
                )

        component_summary[block_name] = {
            "thresholds": thresholds,
            "queue_rows": int(len(block_queue)),
            "reconstruction_error": reconstruction_error,
            "critical_rows": int(
                (block_queue["rank"] <= 3).sum()
            ),
            "elevated_rows": int(
                (
                    (block_queue["rank"] > 3)
                    & (block_queue["rank"] <= 7)
                ).sum()
            ),
            "watch_rows": int(
                (block_queue["rank"] > 7).sum()
            ),
        }
        saved_arrays[f"{block_name}_severity"] = components[
            "severity"
        ]
        saved_arrays[f"{block_name}_memory"] = components["memory"]
        saved_arrays[f"{block_name}_frequency"] = components[
            "frequency"
        ]
        saved_arrays[f"{block_name}_risk"] = components["risk"]
        saved_arrays[f"{block_name}_risk_percentile"] = percentile

    evidence = pd.concat(evidence_frames, ignore_index=True)
    evidence.to_csv(
        OUT / "s3_v2m_s_risk_queue_k10.csv",
        index=False,
        encoding="utf-8",
    )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(
        OUT / "s3_v2m_s_risk_threshold_counts.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "s3_v2m_s_risk_components.npz",
        **saved_arrays,
    )
    summary = {
        "risk_formula": "1.0*severity",
        "risk_mode": "severity_only",
        "window_days": WINDOW_DAYS,
        "decay": DECAY,
        "high_score_threshold": HIGH_SCORE_THRESHOLD,
        "weights": {
            "severity": SEVERITY_WEIGHT,
            "memory": MEMORY_WEIGHT,
            "frequency": FREQUENCY_WEIGHT,
        },
        "warning_thresholds": (
            "training-severity 95/98/99 percentiles"
        ),
        "queue_bands": {
            "critical": "rank 1-3",
            "elevated": "rank 4-7",
            "watch": "rank 8-10",
        },
        "labels_used_in_risk": False,
        "rows": int(len(evidence)),
        "blocks": component_summary,
        "outputs": {
            "queue": str(OUT / "s3_v2m_s_risk_queue_k10.csv"),
            "threshold_counts": str(
                OUT / "s3_v2m_s_risk_threshold_counts.csv"
            ),
            "components": str(
                OUT / "s3_v2m_s_risk_components.npz"
            ),
        },
    }
    (OUT / "s3_v2m_s_risk_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        evidence.groupby(
            ["data_block", "split", "queue_band"]
        ).size().to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()

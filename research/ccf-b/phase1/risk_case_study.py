from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[3]
RUNTIME = REPO / "runtime" / "cert"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    evaluate,
)


SEEDS = tuple(range(20261116, 20261126))
PRIMARY_K = 10
EWMA_ALPHA = 0.2
CUSUM_DRIFT = 0.5
PROPAGATION_BETA = 0.2
TOP_M = 10
HORIZON = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run causal instant/EWMA/CUSUM/propagation risk controls on "
            "the locked 10-seed PAI scores and export S4 case cards."
        )
    )
    parser.add_argument(
        "--cert-data",
        type=Path,
        default=Path(r"E:\MarketRisker\work\cert"),
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=Path(
            r"E:\MarketRisker\work\cert\external"
            r"\pilot_r42_10seed_scores_v1"
        ),
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path(
            r"E:\MarketRisker\work\cert\experiments"
            r"\representation_preservation\e2_instance"
            r"\missing_safe_v2_20261117\soft_instance_labels.npy"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def causal_ewma(
    scores: np.ndarray,
    active: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.full(scores.shape, np.nan, dtype=np.float32)
    state = np.zeros(scores.shape[0], dtype=np.float64)
    for time in range(scores.shape[1]):
        valid = active[:, time] & np.isfinite(scores[:, time])
        state[~active[:, time]] = 0.0
        state[valid] = (
            alpha * scores[valid, time]
            + (1.0 - alpha) * state[valid]
        )
        output[valid, time] = state[valid]
    return output


def causal_cusum(
    scores: np.ndarray,
    active: np.ndarray,
    drift: float,
) -> np.ndarray:
    output = np.full(scores.shape, np.nan, dtype=np.float32)
    state = np.zeros(scores.shape[0], dtype=np.float64)
    for time in range(scores.shape[1]):
        valid = active[:, time] & np.isfinite(scores[:, time])
        state[~active[:, time]] = 0.0
        state[valid] = np.maximum(
            0.0,
            state[valid] + scores[valid, time] - drift,
        )
        output[valid, time] = state[valid]
    return output


def top_m_neighbors(
    graph: np.ndarray,
    top_m: int,
) -> np.ndarray:
    weights = np.asarray(graph, dtype=np.float64).copy()
    if weights.shape[0] != weights.shape[1]:
        raise ValueError("graph must be square")
    np.fill_diagonal(weights, 0.0)
    selected = np.zeros_like(weights)
    for row in range(len(weights)):
        candidates = np.argsort(-weights[row], kind="stable")[:top_m]
        selected[row, candidates] = weights[row, candidates]
    selected[selected < 0.0] = 0.0
    row_sums = selected.sum(axis=1, keepdims=True)
    return selected / np.where(row_sums > 1e-12, row_sums, 1.0)


def neighbor_score(
    scores: np.ndarray,
    active: np.ndarray,
    graph: np.ndarray,
) -> np.ndarray:
    output = np.full(scores.shape, np.nan, dtype=np.float32)
    for time in range(scores.shape[1]):
        valid = active[:, time] & np.isfinite(scores[:, time])
        source = np.where(valid, scores[:, time], 0.0)
        numerator = graph @ source
        denominator = graph @ valid.astype(np.float64)
        neighbor = numerator / np.where(
            denominator > 1e-12,
            denominator,
            1.0,
        )
        output[valid, time] = neighbor[valid]
    return output


def propagate_scores(
    scores: np.ndarray,
    active: np.ndarray,
    graph: np.ndarray,
    beta: float,
) -> np.ndarray:
    neighbors = neighbor_score(scores, active, graph)
    output = scores.copy()
    valid = active & np.isfinite(scores)
    output[valid] = (
        (1.0 - beta) * scores[valid]
        + beta * neighbors[valid]
    )
    return output.astype(np.float32)


def point_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, float]:
    split_score = scores[:, start : end + 1]
    split_label = labels[:, start : end + 1]
    split_active = active[:, start : end + 1]
    valid = split_active & np.isfinite(split_score)
    y_true = split_label[valid].astype(np.int8)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    return (
        float(average_precision_score(y_true, split_score[valid])),
        float(roc_auc_score(y_true, split_score[valid])),
    )


def ranking_components(
    scores: np.ndarray,
    active: np.ndarray,
    graph: np.ndarray,
) -> dict[str, np.ndarray]:
    propagated = propagate_scores(
        scores,
        active,
        graph,
        PROPAGATION_BETA,
    )
    return {
        "instant": scores,
        "ewma": causal_ewma(scores, active, EWMA_ALPHA),
        "cusum": causal_cusum(scores, active, CUSUM_DRIFT),
        "propagation": propagated,
        "ewma_propagation": causal_ewma(
            propagated,
            active,
            EWMA_ALPHA,
        ),
    }


def incident_details(
    alerts: np.ndarray,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    incidents: pd.DataFrame,
    start_index: int,
    end_index: int,
    horizon: int,
) -> pd.DataFrame:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
    }
    rows = []
    for incident in incidents.itertuples(index=False):
        accounts = [
            account
            for account in str(incident.observed_accounts).split("|")
            if account in user_index
        ]
        if not accounts:
            continue
        start_date = pd.Timestamp(incident.start).normalize()
        end_date = pd.Timestamp(incident.end).normalize()
        start_position = date_index.get(start_date.strftime("%Y-%m-%d"))
        if (
            start_position is None
            or start_position < start_index
            or start_position > end_index
        ):
            continue
        candidate_dates = pd.date_range(
            start_date,
            min(end_date, start_date + pd.Timedelta(days=horizon - 1)),
            freq="D",
        )
        first_hit = None
        for date in candidate_dates:
            index = date_index.get(date.strftime("%Y-%m-%d"))
            if index is None or index < start_index or index > end_index:
                continue
            indices = np.asarray(
                [user_index[account] for account in accounts],
                dtype=np.int64,
            )
            if alerts[indices, index].any():
                first_hit = date
                break
        rows.append(
            {
                "incident_id": str(incident.incident_id),
                "scenario": str(incident.scenario),
                "observed_accounts": "|".join(accounts),
                "start_date": start_date,
                "end_date": end_date,
                "hit": first_hit is not None,
                "first_hit_date": first_hit,
                "delay_days": (
                    (first_hit - start_date).days
                    if first_hit is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def select_cases(
    details: pd.DataFrame,
) -> pd.DataFrame:
    ordered = details.sort_values("incident_id", kind="stable")
    successes = ordered[ordered["hit"]].head(2)
    failures = ordered[~ordered["hit"]].head(1)
    return pd.concat([successes, failures], ignore_index=True)


def evidence_rows(
    cases: pd.DataFrame,
    alerts: np.ndarray,
    components: dict[str, np.ndarray],
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    features: np.ndarray,
) -> list[dict[str, object]]:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
    }
    rows = []
    for case in cases.itertuples(index=False):
        accounts = [
            account
            for account in str(case.observed_accounts).split("|")
            if account in user_index
        ]
        if not accounts:
            continue
        start = date_index[pd.Timestamp(case.start_date).strftime("%Y-%m-%d")]
        best_user = user_index[accounts[0]]
        best_time = start
        first_alert_time = None
        for account in accounts:
            user = user_index[account]
            candidate = np.flatnonzero(
                alerts[
                    user,
                    start : min(start + HORIZON, alerts.shape[1]),
                ]
            )
            if len(candidate):
                candidate_time = start + int(candidate[0])
                if first_alert_time is None or candidate_time < first_alert_time:
                    first_alert_time = candidate_time
                    best_user = user
                    best_time = candidate_time
        user = best_user
        time = best_time
        account = str(users[user])
        if first_alert_time is not None:
            alert_date = dates[time]
        else:
            alert_date = None
        component_values = {
            name: (
                None
                if not np.isfinite(values[user, time])
                else float(values[user, time])
            )
            for name, values in components.items()
        }
        raw = features[user, time]
        top_channels = np.argsort(-np.nan_to_num(raw, nan=-np.inf))[:3]
        rows.append(
            {
                "incident_id": str(case.incident_id),
                "observed_account": account,
                "start_date": pd.Timestamp(case.start_date).strftime(
                    "%Y-%m-%d"
                ),
                "first_alert_date": (
                    alert_date.strftime("%Y-%m-%d")
                    if alert_date is not None
                    else None
                ),
                "hit": bool(case.hit),
                "delay_days": case.delay_days,
                **component_values,
                "top_feature_indices": "|".join(
                    str(int(value)) for value in top_channels
                ),
                "top_feature_values": "|".join(
                    f"{float(raw[value]):.6g}" for value in top_channels
                ),
            }
        )
    return rows


def markdown_cases(rows: list[dict[str, object]]) -> str:
    lines = [
        "# S4 controlled-risk case cards",
        "",
        "Selection rule: sort incident IDs lexicographically; take the first",
        "two 3-day hits and the first 3-day miss. This rule is fixed before",
        "inspection and is not changed after seeing scores.",
        "",
    ]
    if not rows:
        lines.append("No eligible cases were found.")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.extend(
            [
                f"## {row['incident_id']}",
                "",
                f"- observed account: `{row['observed_account']}`",
                f"- incident start: `{row['start_date']}`",
                f"- first alert: `{row['first_alert_date']}`",
                f"- hit within 3 days: `{row['hit']}`",
                f"- delay: `{_display_value(row['delay_days'])}`",
                f"- instant risk: `{row['instant']}`",
                f"- EWMA+propagation risk: `{row['ewma_propagation']}`",
                f"- top standardized feature indices: "
                f"`{row['top_feature_indices']}`",
                f"- top standardized feature values: "
                f"`{row['top_feature_values']}`",
                "",
                "The card is ranking evidence, not a causal claim.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _display_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = args.cert_data / "processed_r42"
    with np.load(
        processed / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    ) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        values = dataset["values"].astype(np.float32)
        active = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        val_end = int(dataset["val_end_index"])
        test_start = int(dataset["test_start_index"])
        test_end = int(dataset["test_end_index"])
    labels = np.zeros(active.shape, dtype=np.uint8)
    label_frame = pd.read_csv(
        processed / "r4.2_malicious_account_days.csv",
        parse_dates=["date"],
    )
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
    }
    for row in label_frame.itertuples(index=False):
        if row.user_id in user_index:
            labels[
                user_index[row.user_id],
                date_index[row.date.strftime("%Y-%m-%d")],
            ] = 1
    incidents = pd.read_csv(processed / "r4.2_incidents.csv")
    graph = top_m_neighbors(np.load(args.graph), TOP_M)

    rows: list[dict[str, object]] = []
    case_seed = args.seeds[0]
    case_payload = None
    for seed in args.seeds:
        score_path = (
            args.score_root / f"seed_{seed}" / "A1-Cov_scores.npz"
        )
        with np.load(score_path, allow_pickle=False) as stored:
            base = stored["pai_a1"].astype(np.float32)
            arrays = {
                "pai_a1": base,
                "plain_l1": stored["plain_l1"].astype(np.float32),
                "euc_z": stored["euc_z"].astype(np.float32),
                "mag_g_z": stored["mag_g_z"].astype(np.float32),
                "t2_z": stored["t2_z"].astype(np.float32),
            }
        components = ranking_components(base, active, graph)
        for method, score in components.items():
            alerts = build_alerts(score, active, PRIMARY_K, 7)
            result = evaluate(
                users,
                dates,
                active,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                method,
                PRIMARY_K,
            )
            test = result["splits"]["test"]
            pr_auc, roc_auc = point_metrics(
                score,
                labels,
                active,
                test_start,
                test_end,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "event_recall_3d": test["horizons"][str(HORIZON)][
                        "event_recall"
                    ],
                    "events_hit": test["horizons"][str(HORIZON)][
                        "incidents_hit"
                    ],
                    "events_total": test["horizons"][str(HORIZON)][
                        "incidents_total"
                    ],
                    "median_contact_time_days": test["horizons"][
                        str(HORIZON)
                    ]["median_contact_time_days"],
                    "precision_at_k": test["precision_at_k"],
                    "false_alerts_per_day": test["false_alerts_per_day"],
                }
            )
            if seed == case_seed and method == "ewma_propagation":
                details = incident_details(
                    alerts,
                    users,
                    dates,
                    incidents,
                    test_start,
                    test_end,
                    HORIZON,
                )
                cases = select_cases(details)
                case_payload = evidence_rows(
                    cases,
                    alerts,
                    {
                        **arrays,
                        "instant": components["instant"],
                        "ewma_propagation": score,
                    },
                    users,
                    dates,
                    values,
                )
        print(f"seed={seed} risk controls done", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        args.output_dir / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            events_hit_mean=("events_hit", "mean"),
            median_contact_time_days=("median_contact_time_days", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            false_alerts_per_day_mean=("false_alerts_per_day", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(
        args.output_dir / "summary.csv",
        index=False,
        encoding="utf-8",
    )
    paired_rows = []
    instant = metrics[metrics["method"] == "instant"].set_index("seed")
    for method in sorted(metrics["method"].unique()):
        if method == "instant":
            continue
        treatment = metrics[metrics["method"] == method].set_index("seed")
        common = treatment.index.intersection(instant.index)
        for metric_name in (
            "pr_auc",
            "roc_auc",
            "event_recall_3d",
            "median_contact_time_days",
            "precision_at_k",
            "false_alerts_per_day",
        ):
            delta = (
                treatment.loc[common, metric_name]
                - instant.loc[common, metric_name]
            )
            paired_rows.append(
                {
                    "method": method,
                    "control": "instant",
                    "metric": metric_name,
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)),
                    "positive_seeds": int((delta > 0).sum()),
                    "negative_seeds": int((delta < 0).sum()),
                    "n": int(len(delta)),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        args.output_dir / "paired_vs_instant.csv",
        index=False,
        encoding="utf-8",
    )
    if case_payload is not None:
        pd.DataFrame(case_payload).to_csv(
            args.output_dir / "s4_case_cards.csv",
            index=False,
            encoding="utf-8",
        )
        (args.output_dir / "S4_CASES_generated.md").write_text(
            markdown_cases(case_payload),
            encoding="utf-8",
        )
    protocol = {
        "seeds": args.seeds,
        "primary_k": PRIMARY_K,
        "cooldown_days": 7,
        "ewma_alpha": EWMA_ALPHA,
        "cusum_drift": CUSUM_DRIFT,
        "propagation_beta": PROPAGATION_BETA,
        "top_m": TOP_M,
        "horizon_days": HORIZON,
        "graph_source": str(args.graph),
        "score_source": str(args.score_root / "seed_*/A1-Cov_scores.npz"),
        "risk_labels_used": False,
        "case_selection_rule": (
            "sort by incident_id; first two hits and first miss under "
            "EWMA+propagation for the first configured seed"
        ),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "summary": summary.to_dict(orient="records"),
                "paired_vs_instant": paired.to_dict(orient="records"),
                "cases": case_payload or [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

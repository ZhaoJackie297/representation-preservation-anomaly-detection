from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from audit_clue import iter_json_lines, parse_iso_timestamp


def load_labels(path: Path) -> dict[str, tuple[str, date]]:
    labels: dict[str, tuple[str, date]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["source_uid"]] = (
                row["replacement_uid"],
                date.fromisoformat(row["change_date"]),
            )
    return labels


def select_users(
    user_stats: dict[str, Any],
    labels: dict[str, tuple[str, date]],
    *,
    max_users: int,
    min_events: int,
) -> list[str]:
    required = set(labels) | {
        replacement for replacement, _ in labels.values()
    }
    selected = set(required)
    candidates = sorted(
        (
            (values["events"], uid)
            for uid, values in user_stats.items()
            if values["events"] >= min_events and uid not in selected
        ),
        reverse=True,
    )
    for _, uid in candidates:
        if len(selected) >= max_users:
            break
        selected.add(uid)
    if len(selected) > max_users:
        raise ValueError(
            f"labels require {len(selected)} users, above max_users={max_users}"
        )
    return sorted(selected)


def effective_uid(
    source_uid: str,
    event_date: date,
    labels: dict[str, tuple[str, date]],
) -> str:
    switch = labels.get(source_uid)
    if switch is not None and event_date >= switch[1]:
        return switch[0]
    return source_uid


def build_user_days(
    *,
    input_path: Path,
    audit_path: Path,
    user_stats_path: Path,
    labels_path: Path,
    output_dir: Path,
    train_end: date,
    eval_end: date,
    max_users: int,
    min_events: int,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    user_stats = json.loads(user_stats_path.read_text(encoding="utf-8"))
    labels = load_labels(labels_path)
    selected_users = select_users(
        user_stats,
        labels,
        max_users=max_users,
        min_events=min_events,
    )
    user_index = {uid: index for index, uid in enumerate(selected_users)}

    start_date = date.fromisoformat(audit["first_timestamp"][:10])
    dates = [
        date.fromordinal(ordinal)
        for ordinal in range(
            start_date.toordinal(), eval_end.toordinal() + 1
        )
    ]
    date_index = {day: index for index, day in enumerate(dates)}

    event_types = sorted(audit["event_type_counts"])
    feature_names = [
        *event_types,
        "total_events",
        "distinct_event_types",
        "external_ip_events",
    ]
    values = np.full(
        (len(selected_users), len(dates), len(feature_names)),
        np.nan,
        dtype=np.float32,
    )
    active = np.zeros((len(selected_users), len(dates)), dtype=bool)
    event_type_index = {
        name: index for index, name in enumerate(event_types)
    }
    total_index = len(event_types)
    distinct_counts: dict[tuple[int, int], set[str]] = {}

    for event in iter_json_lines(input_path):
        timestamp = parse_iso_timestamp(str(event["time"]))
        event_date = timestamp.date()
        if event_date not in date_index:
            continue
        source_uid = str(event["uid"])
        uid = effective_uid(source_uid, event_date, labels)
        user_index_value = user_index.get(uid)
        if user_index_value is None:
            continue
        day_index = date_index[event_date]
        event_type = str(event["type"])
        feature_index = event_type_index[event_type]
        if not active[user_index_value, day_index]:
            active[user_index_value, day_index] = True
            values[user_index_value, day_index, :] = 0.0
        values[user_index_value, day_index, feature_index] += 1.0
        values[user_index_value, day_index, total_index] += 1.0
        distinct_counts.setdefault(
            (user_index_value, day_index), set()
        ).add(event_type)
        if event.get("isLocalIP") is False:
            values[user_index_value, day_index, -1] += 1.0

    for (user_index_value, day_index), actions in distinct_counts.items():
        values[
            user_index_value, day_index, total_index + 1
        ] = float(len(actions))

    train_cells = active[:, : date_index[train_end] + 1]
    train_values = values[:, : date_index[train_end] + 1]
    mean = np.nanmean(train_values[train_cells], axis=0)
    std = np.nanstd(train_values[train_cells], axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clue_user_day_dataset.npz"
    np.savez_compressed(
        output_path,
        users=np.asarray(selected_users, dtype="U128"),
        dates=np.asarray(
            [day.isoformat() for day in dates], dtype="U10"
        ),
        features=np.asarray(feature_names, dtype="U80"),
        values=normalized.astype(np.float32),
        active_mask=active,
        train_end_index=np.asarray(date_index[train_end]),
        eval_start_index=np.asarray(date_index[train_end] + 1),
        eval_end_index=np.asarray(date_index[eval_end]),
        mean=mean,
        std=std,
    )
    metadata = {
        "array_file": str(output_path),
        "shape": list(normalized.shape),
        "users": len(selected_users),
        "dates": len(dates),
        "features": feature_names,
        "train_end": train_end.isoformat(),
        "eval_start": date.fromordinal(
            train_end.toordinal() + 1
        ).isoformat(),
        "eval_end": eval_end.isoformat(),
        "selected_users": selected_users,
        "switch_labels": {
            uid: {
                "replacement_uid": replacement,
                "change_date": change.isoformat(),
            }
            for uid, (replacement, change) in labels.items()
        },
        "normalization": "training-period active-cell mean/std",
        "labels_used_for_features": False,
        "switch_labels_used_for_uid_mapping": True,
    }
    metadata_path = output_dir / "clue_user_day_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--user-stats", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--min-events", type=int, default=100)
    args = parser.parse_args()

    metadata = build_user_days(
        input_path=args.input,
        audit_path=args.audit,
        user_stats_path=args.user_stats,
        labels_path=args.labels,
        output_dir=args.output_dir,
        train_end=date.fromisoformat(args.train_end),
        eval_end=date.fromisoformat(args.eval_end),
        max_users=args.max_users,
        min_events=args.min_events,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

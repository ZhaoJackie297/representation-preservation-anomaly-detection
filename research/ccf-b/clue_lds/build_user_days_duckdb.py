from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def load_event_types(audit_path: Path) -> list[str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return sorted(audit["event_type_counts"])


def create_switch_table(
    con: duckdb.DuckDBPyConnection,
    labels_path: Path,
    table_name: str = "clue_switch_map",
) -> None:
    escaped = str(labels_path).replace("'", "''")
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            CAST(source_uid AS VARCHAR) AS source_uid,
            CAST(replacement_uid AS VARCHAR) AS replacement_uid,
            CAST(change_date AS DATE) AS change_date
        FROM read_csv_auto(
            '{escaped}',
            header = true,
            all_varchar = false
        )
        """
    )


def build_user_day_arrays(
    con: duckdb.DuckDBPyConnection,
    *,
    audit_path: Path,
    labels_path: Path,
    output_dir: Path,
    train_start: date,
    train_end: date,
    eval_end: date,
    max_users: int,
    update_events_table: str = "clue_events",
    switch_table: str = "clue_switch_map",
) -> dict[str, Any]:
    create_switch_table(con, labels_path, switch_table)
    event_types = load_event_types(audit_path)
    if max_users <= 0:
        raise ValueError("max_users must be positive")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE selected_users AS
        WITH label_users AS (
            SELECT DISTINCT source_uid AS uid FROM {switch_table}
            UNION
            SELECT DISTINCT replacement_uid AS uid FROM {switch_table}
        ),
        ranked AS (
            SELECT e.uid, count(*) AS events,
                   row_number() OVER (ORDER BY count(*) DESC, e.uid) AS rank
            FROM {update_events_table} e
            WHERE e.uid_type = 'name'
              AND e.event_date BETWEEN DATE '{train_start.isoformat()}'
                                   AND DATE '{train_end.isoformat()}'
              AND e.uid NOT IN (SELECT uid FROM label_users)
            GROUP BY e.uid
        ),
        fillers AS (
            SELECT uid FROM ranked
            WHERE rank <= {max_users - 1}
        )
        SELECT uid FROM label_users
        UNION
        SELECT uid FROM fillers
        """
    )
    selected_count = int(
        con.execute("SELECT count(*) FROM selected_users").fetchone()[0]
    )
    if selected_count > max_users:
        raise ValueError(
            f"label users require {selected_count}, above max_users={max_users}"
        )

    if train_start > train_end or train_end >= eval_end:
        raise ValueError(
            "require train_start <= train_end < eval_end"
        )
    dates = [
        date.fromordinal(value)
        for value in range(
            train_start.toordinal(), eval_end.toordinal() + 1
        )
    ]
    date_index = {day: index for index, day in enumerate(dates)}

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE effective_events AS
        SELECT
            e.timestamp_utc,
            e.event_date,
            COALESCE(s.replacement_uid, e.uid) AS effective_uid,
            e.uid AS source_uid,
            e.event_type,
            e.is_local_ip
        FROM {update_events_table} e
        LEFT JOIN {switch_table} s
          ON e.uid = s.source_uid
         AND e.event_date >= s.change_date
        WHERE e.uid_type = 'name'
          AND e.event_date BETWEEN DATE '{train_start.isoformat()}'
                               AND DATE '{eval_end.isoformat()}'
          AND COALESCE(s.replacement_uid, e.uid) IN (
              SELECT uid FROM selected_users
          )
        """
    )

    conditional_features = ",\n".join(
        "SUM(CASE WHEN event_type = "
        f"'{event_type.replace(chr(39), chr(39) * 2)}' "
        f"THEN 1 ELSE 0 END) AS \"{event_type}\""
        for event_type in event_types
    )
    rows = con.execute(
        f"""
        SELECT
            effective_uid,
            event_date,
            {conditional_features},
            count(*) AS total_events,
            count(DISTINCT event_type) AS distinct_event_types,
            count(*) FILTER (WHERE is_local_ip = false) AS external_ip_events
        FROM effective_events
        GROUP BY effective_uid, event_date
        ORDER BY effective_uid, event_date
        """
    ).fetchall()

    users = [
        row[0]
        for row in con.execute(
            "SELECT uid FROM selected_users ORDER BY uid"
        ).fetchall()
    ]
    user_index = {uid: index for index, uid in enumerate(users)}
    feature_names = [
        *event_types,
        "total_events",
        "distinct_event_types",
        "external_ip_events",
    ]
    values = np.full(
        (len(users), len(dates), len(feature_names)),
        np.nan,
        dtype=np.float32,
    )
    active = np.zeros((len(users), len(dates)), dtype=bool)
    for row in rows:
        uid, event_day, *feature_values = row
        user_row = user_index[uid]
        day_row = date_index[event_day]
        values[user_row, day_row, :] = np.asarray(
            feature_values, dtype=np.float32
        )
        active[user_row, day_row] = True

    train_end_index = date_index[train_end]
    train_values = values[:, : train_end_index + 1]
    train_active = active[:, : train_end_index + 1]
    mean = np.nanmean(train_values[train_active], axis=0)
    std = np.nanstd(train_values[train_active], axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clue_user_day_dataset.npz"
    switch_rows = con.execute(
        f"""
        SELECT source_uid, replacement_uid, change_date
        FROM {switch_table}
        ORDER BY source_uid
        """
    ).fetchall()
    event_id = np.full(active.shape, -1, dtype=np.int16)
    event_metadata = []
    for index, (_, replacement, change_day) in enumerate(switch_rows):
        replacement_row = user_index[replacement]
        start_index = date_index[change_day]
        event_id[replacement_row, start_index:] = index
        event_metadata.append(
            {
                "event_id": index,
                "target_uid": replacement,
                "start_date": change_day.isoformat(),
                "end_date": eval_end.isoformat(),
                "source_uid": switch_rows[index][0],
            }
        )
    metadata = {
        "array_file": str(output_path),
        "shape": list(normalized.shape),
        "users": len(users),
        "dates": len(dates),
        "features": feature_names,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "eval_start": date.fromordinal(
            train_end.toordinal() + 1
        ).isoformat(),
        "eval_end": eval_end.isoformat(),
        "selected_users": users,
        "switch_labels": [
            {
                "source_uid": source,
                "replacement_uid": replacement,
                "change_date": str(change_day),
            }
            for source, replacement, change_day in switch_rows
        ],
        "event_count": len(event_metadata),
        "events": event_metadata,
        "normalization": "training-period active-cell mean/std",
        "labels_used_for_features": False,
        "switch_labels_used_for_uid_mapping": True,
    }
    np.savez_compressed(
        output_path,
        users=np.asarray(users, dtype="U128"),
        dates=np.asarray(
            [day.isoformat() for day in dates], dtype="U10"
        ),
        features=np.asarray(feature_names, dtype="U80"),
        values=normalized.astype(np.float32),
        active_mask=active,
        event_id=event_id,
        train_start_index=np.asarray(0),
        train_end_index=np.asarray(train_end_index),
        eval_start_index=np.asarray(train_end_index + 1),
        eval_end_index=np.asarray(date_index[eval_end]),
        mean=mean,
        std=std,
    )
    (output_dir / "clue_user_day_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database))
    con.execute(f"PRAGMA threads={int(args.threads)}")
    metadata = build_user_day_arrays(
        con,
        audit_path=args.audit,
        labels_path=args.labels,
        output_dir=args.output_dir,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        eval_end=date.fromisoformat(args.eval_end),
        max_users=args.max_users,
    )
    con.close()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

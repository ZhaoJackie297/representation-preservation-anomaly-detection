from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def create_events_table(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    table_name: str = "clue_events",
) -> None:
    escaped = str(input_path).replace("'", "''")
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            CAST("time" AS TIMESTAMP) AS timestamp_utc,
            CAST(CAST("time" AS TIMESTAMP) AS DATE) AS event_date,
            CAST("uid" AS VARCHAR) AS uid,
            CAST("uidType" AS VARCHAR) AS uid_type,
            CAST("type" AS VARCHAR) AS event_type,
            CAST("role" AS VARCHAR) AS role,
            TRY_CAST("isLocalIP" AS BOOLEAN) AS is_local_ip
        FROM read_json(
            '{escaped}',
            format = 'newline_delimited',
            columns = {{
                'time': 'VARCHAR',
                'uid': 'VARCHAR',
                'uidType': 'VARCHAR',
                'type': 'VARCHAR',
                'role': 'VARCHAR',
                'isLocalIP': 'BOOLEAN',
                'params': 'JSON'
            }},
            ignore_errors = true
        )
        """
    )


def grouped_counts(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
) -> dict[str, int]:
    rows = con.execute(
        f"""
        SELECT COALESCE(CAST({column} AS VARCHAR), '<missing>') AS value,
               count(*) AS count
        FROM {table_name}
        GROUP BY 1
        ORDER BY count DESC, value
        """
    ).fetchall()
    return {str(value): int(count) for value, count in rows}


def audit_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str = "clue_events",
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        total_events,
        first_timestamp,
        last_timestamp,
        unique_users,
        unique_dates,
        user_days,
    ) = con.execute(
        f"""
        SELECT count(*),
               min(timestamp_utc),
               max(timestamp_utc),
               count(DISTINCT uid),
               count(DISTINCT event_date),
               count(DISTINCT (uid, event_date))
        FROM {table_name}
        """
    ).fetchone()

    top_users = con.execute(
        f"""
        SELECT uid, count(*) AS events
        FROM {table_name}
        GROUP BY uid
        ORDER BY events DESC, uid
        LIMIT 50
        """
    ).fetchall()
    audit = {
        "total_events": int(total_events),
        "first_timestamp": (
            first_timestamp.isoformat() if first_timestamp else None
        ),
        "last_timestamp": (
            last_timestamp.isoformat() if last_timestamp else None
        ),
        "unique_users": int(unique_users),
        "unique_dates": int(unique_dates),
        "user_days": int(user_days),
        "uid_type_counts": grouped_counts(con, table_name, "uid_type"),
        "event_type_counts": grouped_counts(con, table_name, "event_type"),
        "role_counts": grouped_counts(con, table_name, "role"),
        "local_ip_counts": grouped_counts(con, table_name, "is_local_ip"),
        "top_users_by_events": [
            {"uid": uid, "events": int(events)}
            for uid, events in top_users
        ],
        "dates": [
            str(row[0])
            for row in con.execute(
                f"""
                SELECT DISTINCT event_date
                FROM {table_name}
                ORDER BY event_date
                """
            ).fetchall()
        ],
    }

    user_rows = con.execute(
        f"""
        SELECT uid,
               count(*) AS events,
               list(DISTINCT event_type ORDER BY event_type) AS actions,
               list(DISTINCT event_date ORDER BY event_date) AS active_days
        FROM {table_name}
        WHERE uid_type = 'name'
        GROUP BY uid
        ORDER BY uid
        """
    ).fetchall()
    user_stats = {
        uid: {
            "events": int(events),
            "actions": list(actions),
            "active_days": [str(day) for day in active_days],
        }
        for uid, events, actions, active_days in user_rows
    }
    return audit, user_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-stats-output", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    database = str(args.database) if args.database else ":memory:"
    con = duckdb.connect(database)
    con.execute(f"PRAGMA threads={int(args.threads)}")
    create_events_table(con, args.input)
    audit, user_stats = audit_table(con)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.user_stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.user_stats_output.write_text(
        json.dumps(user_stats, ensure_ascii=False),
        encoding="utf-8",
    )
    con.close()


if __name__ == "__main__":
    main()

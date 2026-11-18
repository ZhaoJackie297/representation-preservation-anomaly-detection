from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb

import audit_r42 as base
import audit_r62 as details


DATA = base.ROOT / "r6.2_extracted" / "r6.2"
OUT_JSON = base.ROOT / "r6.2_audit.json"

EVENT_FILES = {
    "logon.csv": {
        "id": "VARCHAR",
        "date": "VARCHAR",
        "user": "VARCHAR",
        "pc": "VARCHAR",
        "activity": "VARCHAR",
    },
    "device.csv": {
        "id": "VARCHAR",
        "date": "VARCHAR",
        "user": "VARCHAR",
        "pc": "VARCHAR",
        "file_tree": "VARCHAR",
        "activity": "VARCHAR",
    },
    "file.csv": {
        "id": "VARCHAR",
        "date": "VARCHAR",
        "user": "VARCHAR",
        "pc": "VARCHAR",
        "filename": "VARCHAR",
        "activity": "VARCHAR",
        "to_removable_media": "VARCHAR",
        "from_removable_media": "VARCHAR",
        "content": "VARCHAR",
    },
    "email.csv": {
        "id": "VARCHAR",
        "date": "VARCHAR",
        "user": "VARCHAR",
        "pc": "VARCHAR",
        "to": "VARCHAR",
        "cc": "VARCHAR",
        "bcc": "VARCHAR",
        "from": "VARCHAR",
        "activity": "VARCHAR",
        "size": "VARCHAR",
        "attachments": "VARCHAR",
        "content": "VARCHAR",
    },
    "http.csv": {
        "id": "VARCHAR",
        "date": "VARCHAR",
        "user": "VARCHAR",
        "pc": "VARCHAR",
        "url": "VARCHAR",
        "activity": "VARCHAR",
        "content": "VARCHAR",
    },
}


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def audit_event_file(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    columns: dict[str, str],
) -> dict:
    expected_columns = list(columns)
    with path.open("rb") as handle:
        header = (
            handle.readline()
            .decode("utf-8-sig")
            .strip()
            .split(",")
        )
    if header != expected_columns:
        raise RuntimeError(f"{path.name} header mismatch: {header}")

    normalized_path = str(path).replace("\\", "/")
    column_sql = ", ".join(
        f"'{name}': '{kind}'" for name, kind in columns.items()
    )
    source = (
        f"read_csv('{normalized_path}', header=true, "
        f"columns={{{column_sql}}}, ignore_errors=true, "
        "sample_size=1, parallel=true)"
    )
    has_activity = "activity" in columns
    group_clause = (
        "GROUP BY GROUPING SETS ((activity), ())"
        if has_activity
        else ""
    )
    grouping_select = (
        "CASE WHEN grouping(activity) = 1 THEN 1 ELSE 0 END "
        "AS is_total, activity,"
        if has_activity
        else "1 AS is_total, NULL AS activity,"
    )
    query = f"""
        SELECT
          {grouping_select}
          count(*) AS rows,
          count(DISTINCT "user") AS unique_users,
          count(DISTINCT "pc") AS unique_pcs,
          CAST(min(try_strptime("date", '%m/%d/%Y %H:%M:%S')) AS DATE)
            AS date_min,
          CAST(max(try_strptime("date", '%m/%d/%Y %H:%M:%S')) AS DATE)
            AS date_max,
          count(*) FILTER (
            WHERE trim(coalesce("date", '')) = ''
              OR trim(coalesce("user", '')) = ''
              OR trim(coalesce("pc", '')) = ''
          ) AS bad_rows,
          count(*) FILTER (
            WHERE trim(coalesce("date", '')) = ''
          ) AS blank_date,
          count(*) FILTER (
            WHERE trim(coalesce("user", '')) = ''
          ) AS blank_user,
          count(*) FILTER (
            WHERE trim(coalesce("pc", '')) = ''
          ) AS blank_pc,
          count(*) FILTER (
            WHERE try_strptime("date", '%m/%d/%Y %H:%M:%S') IS NULL
          ) AS invalid_date
        FROM {source}
        {group_clause}
    """
    rows = connection.execute(query).fetchall()
    names = [item[0] for item in connection.description]
    records = [dict(zip(names, row)) for row in rows]
    overall = next(record for record in records if record["is_total"] == 1)
    activities = Counter()
    if has_activity:
        activities = Counter(
            {
                str(record["activity"]): int(record["rows"])
                for record in records
                if record["is_total"] == 0
                and record["activity"] is not None
            }
        )

    return {
        "rows": int(overall["rows"]),
        "columns": expected_columns,
        "unique_users": int(overall["unique_users"]),
        "unique_pcs": int(overall["unique_pcs"]),
        "date_min": (
            str(overall["date_min"])
            if overall["date_min"] is not None
            else None
        ),
        "date_max": (
            str(overall["date_max"])
            if overall["date_max"] is not None
            else None
        ),
        "activities": dict(activities),
        "bad_rows": int(overall["bad_rows"]),
        "invalid_date": int(overall["invalid_date"]),
        "blank_early_fields": {
            "date": int(overall["blank_date"]),
            "user": int(overall["blank_user"]),
            "pc": int(overall["blank_pc"]),
        },
    }


def main() -> None:
    base.DATA = DATA
    base.OUT_JSON = OUT_JSON
    summary = {
        "files": {},
        "ldap": base.audit_ldap(),
        "psychometric": base.audit_psychometric(),
        "answers": details.audit_answers_r62(),
    }
    connection = duckdb.connect()
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '4GB'")
    for name, columns in EVENT_FILES.items():
        print(f"Auditing {name} with DuckDB...", flush=True)
        summary["files"][name] = audit_event_file(
            connection, DATA / name, columns
        )
        print(
            json.dumps(summary["files"][name], ensure_ascii=False),
            flush=True,
        )
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

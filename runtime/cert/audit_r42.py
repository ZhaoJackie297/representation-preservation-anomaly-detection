from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "r4.2_extracted" / "r4.2"
ANSWERS = ROOT / "answers.tar.bz2"
OUT_JSON = ROOT / "r4.2_audit.json"

EVENT_FILES = {
    "logon.csv": ["id", "date", "user", "pc", "activity"],
    "device.csv": ["id", "date", "user", "pc", "activity"],
    "file.csv": ["id", "date", "user", "pc", "filename", "content"],
    "email.csv": [
        "id",
        "date",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "size",
        "attachments",
        "content",
    ],
    "http.csv": ["id", "date", "user", "pc", "url", "content"],
}


def day_key(date_text: str) -> str:
    # CSV dates use MM/DD/YYYY HH:MM:SS.
    return date_text[6:10] + "-" + date_text[0:2] + "-" + date_text[3:5]


def audit_event_file(path: Path, expected_columns: list[str]) -> dict:
    rows = 0
    users: set[str] = set()
    pcs: set[str] = set()
    activities = Counter()
    min_day: str | None = None
    max_day: str | None = None
    bad_rows = 0
    blank_early_fields = Counter()
    expected_field_count = len(expected_columns)
    progress_gb = 0

    with path.open("rb", buffering=1024 * 1024) as handle:
        header = handle.readline().decode("utf-8-sig").strip().split(",")
        if header != expected_columns:
            raise RuntimeError(f"{path.name} header mismatch: {header}")

        for raw_line in handle:
            rows += 1
            parts = raw_line.rstrip(b"\r\n").split(b",", 4)
            if len(parts) < min(4, expected_field_count):
                bad_rows += 1
                continue
            date_text = parts[1].decode("utf-8", "replace").strip()
            user = parts[2].decode("utf-8", "replace").strip()
            pc = parts[3].decode("utf-8", "replace").strip()
            if not date_text:
                blank_early_fields["date"] += 1
            if not user:
                blank_early_fields["user"] += 1
            if not pc:
                blank_early_fields["pc"] += 1
            if user:
                users.add(user)
            if pc:
                pcs.add(pc)
            if "activity" in expected_columns and len(parts) > 4:
                activity = parts[4].decode("utf-8", "replace").strip()
                if activity:
                    activities[activity] += 1
            if date_text:
                current_day = day_key(date_text)
                min_day = current_day if min_day is None or current_day < min_day else min_day
                max_day = current_day if max_day is None or current_day > max_day else max_day

            if rows % 5_000_000 == 0:
                print(f"  {path.name}: {rows:,} rows", flush=True)

    return {
        "rows": rows,
        "columns": expected_columns,
        "unique_users": len(users),
        "unique_pcs": len(pcs),
        "date_min": min_day,
        "date_max": max_day,
        "activities": dict(activities),
        "bad_rows": bad_rows,
        "blank_early_fields": dict(blank_early_fields),
    }


def audit_ldap() -> dict:
    per_month = {}
    latest_frame = None
    latest_month = ""
    all_fields = set()
    for path in sorted((DATA / "LDAP").glob("*.csv")):
        month = path.stem
        frame = pd.read_csv(path, dtype="string")
        all_fields.update(frame.columns)
        roles = frame["role"].fillna("Unknown").value_counts()
        departments = frame["department"].fillna("Unknown").value_counts()
        per_month[month] = {
            "rows": int(len(frame)),
            "unique_users": int(frame["user_id"].nunique()),
            "role_count": int(roles.size),
            "department_count": int(departments.size),
        }
        if month > latest_month:
            latest_month = month
            latest_frame = frame
    if latest_frame is None:
        raise RuntimeError("No LDAP files found.")

    return {
        "files": per_month,
        "latest_month": latest_month,
        "fields": sorted(all_fields),
        "latest_users": int(latest_frame["user_id"].nunique()),
        "latest_roles": latest_frame["role"].fillna("Unknown").value_counts().to_dict(),
        "latest_business_units": latest_frame["business_unit"]
        .fillna("Unknown")
        .value_counts()
        .to_dict(),
        "latest_functional_units": latest_frame["functional_unit"]
        .fillna("Unknown")
        .value_counts()
        .to_dict(),
        "latest_departments": latest_frame["department"]
        .fillna("Unknown")
        .value_counts()
        .to_dict(),
        "latest_teams": latest_frame["team"].fillna("Unknown").value_counts().to_dict(),
        "latest_supervisors": int(latest_frame["supervisor"].fillna("Unknown").nunique()),
        "missing_latest": {
            column: int(latest_frame[column].isna().sum())
            for column in latest_frame.columns
        },
    }


def audit_psychometric() -> dict:
    frame = pd.read_csv(DATA / "psychometric.csv", dtype={"user_id": "string"})
    return {
        "rows": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "columns": frame.columns.tolist(),
        "missing": {column: int(frame[column].isna().sum()) for column in frame.columns},
    }


def audit_answers() -> dict:
    with tarfile.open(ANSWERS, "r:bz2") as archive:
        raw = archive.extractfile("answers/insiders.csv")
        if raw is None:
            raise RuntimeError("Missing answers/insiders.csv")
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        rows = [row for row in csv.DictReader(text) if row["dataset"].strip() == "4.2"]

    durations = []
    starts = []
    ends = []
    for row in rows:
        start = datetime.strptime(row["start"].strip(), "%m/%d/%Y %H:%M:%S")
        end = datetime.strptime(row["end"].strip(), "%m/%d/%Y %H:%M:%S")
        starts.append(start)
        ends.append(end)
        durations.append((end - start).total_seconds() / 86400.0)

    rows_sorted = sorted(durations)
    return {
        "rows": len(rows),
        "scenario_counts": dict(Counter(row["scenario"].strip() for row in rows)),
        "unique_users": len({row["user"].strip() for row in rows}),
        "start_min": min(starts).strftime("%Y-%m-%d %H:%M:%S"),
        "start_max": max(starts).strftime("%Y-%m-%d %H:%M:%S"),
        "end_min": min(ends).strftime("%Y-%m-%d %H:%M:%S"),
        "end_max": max(ends).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_days_median": rows_sorted[len(rows_sorted) // 2],
        "duration_days_min": min(rows_sorted),
        "duration_days_max": max(rows_sorted),
    }


def main() -> None:
    summary = {
        "files": {},
        "ldap": audit_ldap(),
        "psychometric": audit_psychometric(),
        "answers": audit_answers(),
    }
    for name, columns in EVENT_FILES.items():
        print(f"Reading {name}...", flush=True)
        summary["files"][name] = audit_event_file(DATA / name, columns)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

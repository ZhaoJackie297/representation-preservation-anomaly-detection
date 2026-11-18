from __future__ import annotations

import csv
import io
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import audit_r42 as base


VERSION = "5.2"
base.DATA = base.ROOT / "r5.2_extracted" / "r5.2"
base.OUT_JSON = base.ROOT / "r5.2_audit.json"
base.EVENT_FILES = {
    "logon.csv": ["id", "date", "user", "pc", "activity"],
    "device.csv": [
        "id",
        "date",
        "user",
        "pc",
        "file_tree",
        "activity",
    ],
    "file.csv": [
        "id",
        "date",
        "user",
        "pc",
        "filename",
        "activity",
        "to_removable_media",
        "from_removable_media",
        "content",
    ],
    "email.csv": [
        "id",
        "date",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "activity",
        "size",
        "attachments",
        "content",
    ],
    "http.csv": ["id", "date", "user", "pc", "url", "content"],
}


def audit_event_file_r52(
    path: Path,
    expected_columns: list[str],
) -> dict:
    rows = 0
    users = set()
    pcs = set()
    activities = Counter()
    min_day = None
    max_day = None
    bad_rows = 0
    blank_early_fields = Counter()

    with path.open("rb", buffering=1024 * 1024) as handle:
        header = (
            handle.readline()
            .decode("utf-8-sig")
            .strip()
            .split(",")
        )
        if header != expected_columns:
            raise RuntimeError(
                f"{path.name} header mismatch: {header}"
            )

        if path.name in {"device.csv", "file.csv"}:
            split_count = 5
            activity_index = 5
        elif path.name == "email.csv":
            split_count = 8
            activity_index = 8
        elif path.name == "http.csv" and "activity" in expected_columns:
            split_count = 5
            activity_index = 5
        else:
            split_count = 4
            activity_index = 4

        for raw_line in handle:
            rows += 1
            parts = raw_line.rstrip(b"\r\n").split(
                b",", split_count
            )
            if len(parts) <= 3:
                bad_rows += 1
                continue
            date_text = parts[1].decode(
                "utf-8", "replace"
            ).strip()
            user = parts[2].decode(
                "utf-8", "replace"
            ).strip()
            pc = parts[3].decode(
                "utf-8", "replace"
            ).strip()
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
            if "activity" in expected_columns and len(parts) > activity_index:
                activity = (
                    parts[activity_index]
                    .split(b",", 1)[0]
                    .decode("utf-8", "replace")
                    .strip()
                )
                if activity:
                    activities[activity] += 1
            if date_text:
                current_day = base.day_key(date_text)
                min_day = (
                    current_day
                    if min_day is None or current_day < min_day
                    else min_day
                )
                max_day = (
                    current_day
                    if max_day is None or current_day > max_day
                    else max_day
                )
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


def audit_answers_r52() -> dict:
    with tarfile.open(base.ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        raw_master = archive.extractfile("answers/insiders.csv")
        if raw_master is None:
            raise RuntimeError("Missing answers/insiders.csv")
        master = list(
            csv.DictReader(
                io.TextIOWrapper(
                    raw_master, encoding="utf-8-sig", newline=""
                )
            )
        )

        selected = [
            row
            for row in master
            if row["dataset"].strip() == VERSION
        ]
        durations = []
        starts = []
        ends = []
        first_events = []
        last_events = []
        observed_accounts = set()
        malicious_account_days = set()
        insider_ids = set()
        event_counts = []
        scenario_counts = Counter()

        for row in selected:
            scenario = row["scenario"].strip()
            insider = row["user"].strip()
            detail = row["details"].strip()
            start = datetime.strptime(
                row["start"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            end = datetime.strptime(
                row["end"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            scenario_counts[scenario] += 1
            insider_ids.add(insider)
            starts.append(start)
            ends.append(end)
            durations.append((end - start).total_seconds() / 86400.0)

            candidates = [
                f"answers/{detail}",
                f"answers/r{VERSION}-{scenario}/{detail}",
            ]
            detail_path = next(
                (candidate for candidate in candidates if candidate in names),
                None,
            )
            if detail_path is None:
                raise FileNotFoundError(detail)
            raw_detail = archive.extractfile(detail_path)
            if raw_detail is None:
                raise RuntimeError(f"Cannot read {detail_path}")
            detail_events = []
            for event in csv.reader(
                io.TextIOWrapper(
                    raw_detail, encoding="utf-8-sig", newline=""
                )
            ):
                if len(event) < 4:
                    continue
                timestamp = datetime.strptime(
                    event[2].strip(), "%m/%d/%Y %H:%M:%S"
                )
                account = event[3].strip()
                detail_events.append(timestamp)
                observed_accounts.add(account)
                malicious_account_days.add((account, timestamp.date()))
            if not detail_events:
                raise RuntimeError(f"No events in {detail_path}")
            first_events.append(min(detail_events))
            last_events.append(max(detail_events))
            event_counts.append(len(detail_events))

    return {
        "rows": len(selected),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "unique_insiders": len(insider_ids),
        "unique_observed_accounts": len(observed_accounts),
        "malicious_account_days": len(malicious_account_days),
        "missing_insider_from_observed_accounts": sorted(
            insider_ids - observed_accounts
        ),
        "start_min": min(starts).strftime("%Y-%m-%d %H:%M:%S"),
        "start_max": max(starts).strftime("%Y-%m-%d %H:%M:%S"),
        "end_min": min(ends).strftime("%Y-%m-%d %H:%M:%S"),
        "end_max": max(ends).strftime("%Y-%m-%d %H:%M:%S"),
        "first_observed_event_min": min(first_events).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "last_observed_event_max": max(last_events).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "detail_events_total": sum(event_counts),
        "detail_events_min": min(event_counts),
        "detail_events_median": sorted(event_counts)[len(event_counts) // 2],
        "detail_events_max": max(event_counts),
        "duration_days_min": min(durations),
        "duration_days_median": sorted(durations)[len(durations) // 2],
        "duration_days_max": max(durations),
    }


base.audit_answers = audit_answers_r52
base.audit_event_file = audit_event_file_r52


if __name__ == "__main__":
    base.main()


from __future__ import annotations

import csv
import io
import tarfile
from collections import Counter
from datetime import datetime

import audit_r42 as base
import audit_r52 as compat


VERSION = "6.2"
base.DATA = base.ROOT / "r6.2_extracted" / "r6.2"
base.OUT_JSON = base.ROOT / "r6.2_audit.json"


def audit_answers_r62() -> dict:
    with tarfile.open(base.ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        raw_master = archive.extractfile("answers/insiders.csv")
        if raw_master is None:
            raise RuntimeError("Missing answers/insiders.csv")
        selected = [
            row
            for row in csv.DictReader(
                io.TextIOWrapper(
                    raw_master, encoding="utf-8-sig", newline=""
                )
            )
            if row["dataset"].strip() == VERSION
        ]

        starts = []
        ends = []
        durations = []
        scenario_counts = Counter()
        insider_ids = set()
        observed_accounts = set()
        malicious_account_days = set()
        event_counts = []
        first_events = []
        last_events = []

        for incident in selected:
            scenario = incident["scenario"].strip()
            insider = incident["user"].strip()
            start_text = incident["start"].strip()
            # The official r6.2 scenario-4 start is missing its leading "02".
            if start_text.startswith("/"):
                start_text = "02" + start_text
            start = datetime.strptime(
                start_text, "%m/%d/%Y %H:%M:%S"
            )
            end = datetime.strptime(
                incident["end"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            detail = incident["details"].strip()
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

            timestamps = []
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
                timestamps.append(timestamp)
                observed_accounts.add(account)
                malicious_account_days.add(
                    (account, timestamp.date())
                )
            if not timestamps:
                raise RuntimeError(f"No events in {detail_path}")
            first_events.append(min(timestamps))
            last_events.append(max(timestamps))
            event_counts.append(len(timestamps))
            scenario_counts[scenario] += 1
            insider_ids.add(insider)
            starts.append(start)
            ends.append(end)
            durations.append((end - start).total_seconds() / 86400.0)

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
        "detail_events_median": sorted(event_counts)[
            len(event_counts) // 2
        ],
        "detail_events_max": max(event_counts),
        "duration_days_min": min(durations),
        "duration_days_median": sorted(durations)[
            len(durations) // 2
        ],
        "duration_days_max": max(durations),
    }


base.EVENT_FILES = compat.base.EVENT_FILES
base.EVENT_FILES["http.csv"] = [
    "id",
    "date",
    "user",
    "pc",
    "url",
    "activity",
    "content",
]
base.audit_event_file = compat.audit_event_file_r52
base.audit_answers = audit_answers_r62


if __name__ == "__main__":
    base.main()

from __future__ import annotations

import csv
import io
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ANSWERS = ROOT / "answers.tar.bz2"
OUTPUT = ROOT / "processed_r62" / "r6.2_incidents.csv"
VERSION = "6.2"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        raw_master = archive.extractfile("answers/insiders.csv")
        if raw_master is None:
            raise RuntimeError("Missing answers/insiders.csv")
        master = [
            row
            for row in csv.DictReader(
                io.TextIOWrapper(
                    raw_master, encoding="utf-8-sig", newline=""
                )
            )
            if row["dataset"].strip() == VERSION
        ]

        rows = []
        for incident in master:
            scenario = incident["scenario"].strip()
            insider_id = incident["user"].strip()
            detail = incident["details"].strip()
            start_text = incident["start"].strip()
            if start_text.startswith("/"):
                start_text = "02" + start_text
            start = datetime.strptime(
                start_text, "%m/%d/%Y %H:%M:%S"
            )
            end = datetime.strptime(
                incident["end"].strip(), "%m/%d/%Y %H:%M:%S"
            )
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

            observed_accounts = set()
            event_times = []
            event_types = {}
            detail_event_count = 0
            for event in csv.reader(
                io.TextIOWrapper(
                    raw_detail, encoding="utf-8-sig", newline=""
                )
            ):
                if len(event) < 4:
                    continue
                event_type = event[0].strip()
                event_time = datetime.strptime(
                    event[2].strip(), "%m/%d/%Y %H:%M:%S"
                )
                account = event[3].strip()
                observed_accounts.add(account)
                event_times.append(event_time)
                event_types[event_type] = event_types.get(event_type, 0) + 1
                detail_event_count += 1

            rows.append(
                {
                    "incident_id": (
                        f"{VERSION}:{scenario}:{insider_id}:"
                        f"{Path(detail).stem}"
                    ),
                    "scenario": scenario,
                    "insider_id": insider_id,
                    "observed_accounts": "|".join(
                        sorted(observed_accounts)
                    ),
                    "start": start,
                    "end": end,
                    "first_observed_date": min(event_times).date(),
                    "last_observed_date": max(event_times).date(),
                    "detail_event_count": detail_event_count,
                    "event_types": "|".join(
                        f"{name}:{count}"
                        for name, count in sorted(event_types.items())
                    ),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["first_observed_date", "incident_id"]
    )
    frame.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote {len(frame)} incidents to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

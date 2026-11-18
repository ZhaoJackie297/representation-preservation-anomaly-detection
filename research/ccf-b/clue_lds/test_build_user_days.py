from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_user_days import build_user_days


class BuildUserDaysTest(unittest.TestCase):
    def test_switch_changes_effective_uid(self) -> None:
        events = [
            {
                "time": "2022-01-01T10:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
                "isLocalIP": False,
            },
            {
                "time": "2022-01-03T10:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
            },
            {
                "time": "2022-01-04T10:00:00Z",
                "uid": "u2",
                "uidType": "name",
                "type": "login_successful",
            },
        ]
        audit = {
            "first_timestamp": "2022-01-01T00:00:00+00:00",
            "last_timestamp": "2022-01-04T00:00:00+00:00",
            "event_type_counts": {
                "file_accessed": 2,
                "login_successful": 1,
            },
        }
        user_stats = {
            "u1": {
                "events": 2,
                "actions": ["file_accessed"],
                "active_days": ["2022-01-01", "2022-01-03"],
            },
            "u2": {
                "events": 1,
                "actions": ["login_successful"],
                "active_days": ["2022-01-04"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clue_path = root / "clue.json"
            clue_path.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            stats_path = root / "stats.json"
            stats_path.write_text(
                json.dumps(user_stats), encoding="utf-8"
            )
            labels_path = root / "labels.csv"
            with labels_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source_uid",
                        "replacement_uid",
                        "change_date",
                        "similarity",
                        "source_events",
                        "replacement_events",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_uid": "u1",
                        "replacement_uid": "u2",
                        "change_date": "2022-01-03",
                        "similarity": 0.5,
                        "source_events": 2,
                        "replacement_events": 1,
                    }
                )
                writer.writerow(
                    {
                        "source_uid": "u2",
                        "replacement_uid": "u1",
                        "change_date": "2022-01-03",
                        "similarity": 0.5,
                        "source_events": 1,
                        "replacement_events": 2,
                    }
                )
            metadata = build_user_days(
                input_path=clue_path,
                audit_path=audit_path,
                user_stats_path=stats_path,
                labels_path=labels_path,
                output_dir=root / "out",
                train_end=date(2022, 1, 2),
                eval_end=date(2022, 1, 4),
                max_users=2,
                min_events=1,
            )
            with np.load(
                root / "out" / "clue_user_day_dataset.npz"
            ) as data:
                users = data["users"].tolist()
                values = data["values"].copy()
                active = data["active_mask"].copy()
                dates = data["dates"].tolist()
                features = data["features"].tolist()
            u2_index = users.index("u2")
            file_index = features.index("file_accessed")
            day_index = dates.index("2022-01-03")
            self.assertTrue(active[u2_index, day_index])
            self.assertEqual(values[u2_index, day_index, file_index], 0.0)
            self.assertEqual(metadata["users"], 2)


if __name__ == "__main__":
    unittest.main()

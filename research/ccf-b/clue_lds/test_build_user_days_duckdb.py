from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np

from audit_clue_duckdb import create_events_table
from build_user_days_duckdb import build_user_day_arrays


class BuildUserDaysDuckDBTest(unittest.TestCase):
    def test_switch_mapping(self) -> None:
        events = [
            {
                "time": "2022-01-01T10:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
                "params": {},
            },
            {
                "time": "2022-01-03T10:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
                "params": {},
            },
            {
                "time": "2022-01-04T10:00:00Z",
                "uid": "u2",
                "uidType": "name",
                "type": "login_successful",
                "params": {},
            },
        ]
        audit = {
            "event_type_counts": {
                "file_accessed": 2,
                "login_successful": 1,
            }
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

            con = duckdb.connect()
            create_events_table(con, clue_path)
            metadata = build_user_day_arrays(
                con,
                audit_path=audit_path,
                labels_path=labels_path,
                output_dir=root / "out",
                train_start=date(2022, 1, 1),
                train_end=date(2022, 1, 2),
                eval_end=date(2022, 1, 4),
                max_users=2,
            )
            con.close()
            with np.load(root / "out" / "clue_user_day_dataset.npz") as data:
                users = data["users"].tolist()
                active = data["active_mask"].copy()
                dates = data["dates"].tolist()
                event_id = data["event_id"].copy()
            u2_index = users.index("u2")
            day_index = dates.index("2022-01-03")
            self.assertTrue(active[u2_index, day_index])
            self.assertGreaterEqual(event_id[u2_index, day_index], 0)
            self.assertEqual(metadata["users"], 2)
            self.assertEqual(metadata["event_count"], 2)


if __name__ == "__main__":
    unittest.main()

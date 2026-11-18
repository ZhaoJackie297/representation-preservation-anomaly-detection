from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from audit_clue import audit_file, parse_iso_timestamp


class AuditClueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            {
                "time": "2019-11-14T11:26:43Z",
                "uid": "u1",
                "uidType": "name",
                "type": "login_successful",
                "params": {"user": "u1"},
            },
            {
                "time": "2019-11-14T11:26:51Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
                "role": "technical",
                "isLocalIP": False,
                "params": {"path": "/a/b"},
            },
            {
                "time": "2019-11-15T08:00:00Z",
                "uid": "u2",
                "uidType": "name",
                "type": "group_member_added",
            },
        ]

    def test_audit_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clue.json"
            path.write_text(
                "\n".join(json.dumps(event) for event in self.events),
                encoding="utf-8",
            )
            audit = audit_file(path, progress_every=0)
        summary = audit.summary()
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["unique_users"], 2)
        self.assertEqual(summary["unique_dates"], 2)
        self.assertEqual(summary["user_days"], 2)
        self.assertEqual(
            summary["event_type_counts"]["file_accessed"], 1
        )
        self.assertEqual(summary["role_counts"]["<missing>"], 2)

    def test_user_stats_keep_actions_and_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clue.json"
            path.write_text(
                "\n".join(json.dumps(event) for event in self.events),
                encoding="utf-8",
            )
            audit = audit_file(path, progress_every=0)
        stats = audit.user_stats()
        self.assertEqual(
            stats["u1"]["actions"],
            ["file_accessed", "login_successful"],
        )
        self.assertEqual(
            stats["u1"]["active_days"], ["2019-11-14"]
        )

    def test_max_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clue.json"
            path.write_text(
                "\n".join(json.dumps(event) for event in self.events),
                encoding="utf-8",
            )
            audit = audit_file(path, max_events=2, progress_every=0)
        self.assertEqual(audit.total_events, 2)

    def test_timestamp_requires_timezone_or_assumes_utc(self) -> None:
        self.assertEqual(
            parse_iso_timestamp("2020-01-01T00:00:00").isoformat(),
            "2020-01-01T00:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()

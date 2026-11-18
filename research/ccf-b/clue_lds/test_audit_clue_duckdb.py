from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import duckdb

from audit_clue_duckdb import audit_table, create_events_table


class AuditClueDuckDBTest(unittest.TestCase):
    def test_audit_table(self) -> None:
        events = [
            {
                "time": "2020-01-01T00:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "login_successful",
                "role": "technical",
                "params": {},
            },
            {
                "time": "2020-01-01T01:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "file_accessed",
                "isLocalIP": False,
                "params": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clue.json"
            path.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )
            con = duckdb.connect()
            create_events_table(con, path)
            audit, user_stats = audit_table(con)
            con.close()
        self.assertEqual(audit["total_events"], 2)
        self.assertEqual(audit["unique_users"], 1)
        self.assertEqual(audit["user_days"], 1)
        self.assertEqual(
            user_stats["u1"]["actions"],
            ["file_accessed", "login_successful"],
        )


if __name__ == "__main__":
    unittest.main()

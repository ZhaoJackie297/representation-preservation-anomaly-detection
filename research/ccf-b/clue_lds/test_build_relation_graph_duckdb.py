from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np

from build_relation_graph_duckdb import (
    build_graph,
    create_relation_events,
)


class RelationGraphTest(unittest.TestCase):
    def test_group_and_share_edges(self) -> None:
        events = [
            {
                "time": "2022-01-01T00:00:00Z",
                "uid": "admin",
                "uidType": "name",
                "type": "user_added_to_group",
                "params": {"user": "u1", "group": "g1"},
            },
            {
                "time": "2022-01-01T00:00:00Z",
                "uid": "admin",
                "uidType": "name",
                "type": "user_added_to_group",
                "params": {"user": "u2", "group": "g1"},
            },
            {
                "time": "2022-01-02T00:00:00Z",
                "uid": "u1",
                "uidType": "name",
                "type": "shared_user",
                "params": {"shareWith": "u3"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clue.json"
            path.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )
            con = duckdb.connect()
            create_relation_events(
                con, path, train_end="2022-01-31"
            )
            graph, metadata = build_graph(
                con,
                users=["u1", "u2", "u3"],
                train_end="2022-01-31",
            )
            con.close()
        self.assertGreater(graph[0, 1], 0)
        self.assertGreater(graph[0, 2], 0)
        self.assertEqual(metadata["isolated_users"], 0)
        self.assertTrue(np.allclose(graph.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()

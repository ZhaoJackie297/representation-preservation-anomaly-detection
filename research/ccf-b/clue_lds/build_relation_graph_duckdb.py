from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


RELATION_TYPES = (
    "user_added_to_group",
    "user_removed_from_group",
    "shared_user",
    "shared_group",
    "unshared_user",
    "unshared_group",
    "permission_changed",
)


def create_relation_events(
    con: duckdb.DuckDBPyConnection,
    raw_path: Path,
    *,
    train_end: str,
    table_name: str = "clue_relation_events",
) -> None:
    escaped = str(raw_path).replace("'", "''")
    allowed = ", ".join(f"'{value}'" for value in RELATION_TYPES)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            CAST(CAST("time" AS TIMESTAMP) AS DATE) AS event_date,
            CAST("type" AS VARCHAR) AS event_type,
            CAST("uid" AS VARCHAR) AS actor_uid,
            json_extract_string("params", '$.user') AS member_uid,
            json_extract_string("params", '$.group') AS group_uid,
            json_extract_string("params", '$.shareWith') AS peer_uid
        FROM read_json(
            '{escaped}',
            format = 'newline_delimited',
            columns = {{
                'time': 'VARCHAR',
                'uid': 'VARCHAR',
                'uidType': 'VARCHAR',
                'type': 'VARCHAR',
                'role': 'VARCHAR',
                'isLocalIP': 'BOOLEAN',
                'params': 'JSON'
            }},
            ignore_errors = true
        )
        WHERE CAST(CAST("time" AS TIMESTAMP) AS DATE)
              <= DATE '{train_end}'
          AND "type" IN ({allowed})
        """
    )


def build_graph(
    con: duckdb.DuckDBPyConnection,
    *,
    users: list[str],
    train_end: str,
    relation_table: str = "clue_relation_events",
) -> tuple[np.ndarray, dict[str, object]]:
    user_index = {uid: index for index, uid in enumerate(users)}
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE group_membership AS
        WITH ranked AS (
            SELECT
                member_uid,
                group_uid,
                event_type,
                row_number() OVER (
                    PARTITION BY member_uid, group_uid
                    ORDER BY event_date DESC
                ) AS rank
            FROM {relation_table}
            WHERE event_type IN (
                'user_added_to_group',
                'user_removed_from_group'
            )
              AND member_uid IS NOT NULL
              AND group_uid IS NOT NULL
        )
        SELECT member_uid, group_uid
        FROM ranked
        WHERE rank = 1 AND event_type = 'user_added_to_group'
        """
    )
    group_edges = con.execute(
        """
        SELECT a.member_uid, b.member_uid, count(*) AS shared_groups
        FROM group_membership a
        JOIN group_membership b
          ON a.group_uid = b.group_uid
         AND a.member_uid < b.member_uid
        GROUP BY 1, 2
        """
    ).fetchall()
    share_edges = con.execute(
        f"""
        SELECT
            least(actor_uid, peer_uid) AS uid_a,
            greatest(actor_uid, peer_uid) AS uid_b,
            count(*) AS interactions
        FROM {relation_table}
        WHERE event_type IN (
            'shared_user',
            'shared_group',
            'unshared_user',
            'unshared_group',
            'permission_changed'
        )
          AND actor_uid IS NOT NULL
          AND peer_uid IS NOT NULL
          AND actor_uid <> peer_uid
        GROUP BY 1, 2
        """
    ).fetchall()

    edge_values: dict[tuple[int, int], list[float]] = {}
    max_groups = max((int(row[2]) for row in group_edges), default=1)
    max_shares = max((int(row[2]) for row in share_edges), default=1)
    for left, right, count in group_edges:
        if left not in user_index or right not in user_index:
            continue
        edge = tuple(sorted((user_index[left], user_index[right])))
        edge_values.setdefault(edge, [0.0, 0.0])[0] = (
            int(count) / max_groups
        )
    for left, right, count in share_edges:
        if left not in user_index or right not in user_index:
            continue
        edge = tuple(sorted((user_index[left], user_index[right])))
        edge_values.setdefault(edge, [0.0, 0.0])[1] = (
            np.log1p(int(count)) / np.log1p(max_shares)
        )

    graph = np.zeros((len(users), len(users)), dtype=np.float32)
    for (left, right), (group_score, share_score) in edge_values.items():
        weight = 0.6 * group_score + 0.4 * share_score
        graph[left, right] = weight
        graph[right, left] = weight
    degree = graph.sum(axis=1)
    nonzero_rows = degree > 0
    graph[nonzero_rows] /= degree[nonzero_rows, None]

    return graph, {
        "users": len(users),
        "train_end": train_end,
        "relation_types": list(RELATION_TYPES),
        "group_edges": len(group_edges),
        "share_edges": len(share_edges),
        "weighting": "0.6_group_cooccurrence + 0.4_log1p_share_interactions",
        "nonzero_edges": len(edge_values),
        "isolated_users": int((degree == 0).sum()),
        "row_normalized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    with np.load(args.dataset) as data:
        users = data["users"].tolist()
    con = duckdb.connect(str(args.database))
    con.execute(f"PRAGMA threads={int(args.threads)}")
    create_relation_events(
        con, args.raw, train_end=args.train_end
    )
    graph, metadata = build_graph(
        con, users=users, train_end=args.train_end
    )
    con.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, graph)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

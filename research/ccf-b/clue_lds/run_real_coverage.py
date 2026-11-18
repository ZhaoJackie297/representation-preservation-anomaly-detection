from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PHASE0 = Path(__file__).resolve().parents[1] / "phase0"
sys.path.insert(0, str(PHASE0))

from common_coverage_sampler import (  # noqa: E402
    build_common_coverage_table,
    pair_jaccard,
)


def run(
    *,
    graph_file: Path,
    seeds: list[int],
    target_coverage: float,
    min_batches: int,
    max_batches: int,
) -> dict[str, object]:
    with np.load(graph_file) as data:
        targets = {
            name: {tuple(map(int, edge)) for edge in data[name]}
            for name in ("A3", "A5a", "A5b")
        }
        user_count = int(data["prior"].shape[0])
    results = []
    for seed in seeds:
        result = build_common_coverage_table(
            user_count=user_count,
            target_edge_sets=targets,
            seed=seed,
            stage=0,
            target_coverage=target_coverage,
            min_training_batches=min_batches,
            max_coverage_batches=max_batches,
        )
        results.append(result)
    return {
        "graph_file": str(graph_file),
        "user_count": user_count,
        "seeds": seeds,
        "target_coverage": target_coverage,
        "min_batches": min_batches,
        "max_batches": max_batches,
        "runs": [result.to_summary() for result in results],
        "pair_jaccard": {
            f"{left.seed}:{right.seed}": pair_jaccard(left, right)
            for left, right in zip(results, results[1:])
        },
        "unique_batch_tables": len(
            {result.batch_table_sha256 for result in results}
        )
        == len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--min-batches", type=int, default=520)
    parser.add_argument("--max-batches", type=int, default=10_000)
    args = parser.parse_args()

    payload = run(
        graph_file=args.graphs,
        seeds=args.seeds,
        target_coverage=args.target_coverage,
        min_batches=args.min_batches,
        max_batches=args.max_batches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

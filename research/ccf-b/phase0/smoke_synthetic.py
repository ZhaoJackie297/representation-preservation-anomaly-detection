from __future__ import annotations

import argparse
import json
import random

from common_coverage_sampler import (
    CoverageError,
    build_common_coverage_table,
    normalize_edge,
    pair_jaccard,
)


def degree_graph(user_count: int, degree: int, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    graph: set[tuple[int, int]] = set()
    degrees = [0] * user_count
    for user in range(user_count):
        while degrees[user] < degree:
            candidate = rng.randrange(user_count - 1)
            if candidate >= user:
                candidate += 1
            edge = normalize_edge(user, candidate)
            if edge in graph:
                continue
            graph.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
    return graph


def permute_graph(
    edges: set[tuple[int, int]], user_count: int, seed: int
) -> set[tuple[int, int]]:
    users = list(range(user_count))
    random.Random(seed).shuffle(users)
    return {normalize_edge(users[u], users[v]) for u, v in edges}


def run_case(
    *,
    user_count: int,
    degree: int,
    seeds: list[int],
    target_coverage: float,
    min_batches: int,
    max_batches: int,
) -> dict[str, object]:
    base = degree_graph(user_count, degree, seed=12345)
    random_graph = degree_graph(user_count, degree, seed=67890)
    permuted = permute_graph(base, user_count, seed=24680)
    target_sets = {
        "A3": base,
        "A5a": random_graph,
        "A5b": permuted,
    }
    results = []
    for seed in seeds:
        result = build_common_coverage_table(
            user_count=user_count,
            target_edge_sets=target_sets,
            seed=seed,
            stage=0,
            target_coverage=target_coverage,
            min_training_batches=min_batches,
            max_coverage_batches=max_batches,
        )
        results.append(result)
    return {
        "config": {
            "user_count": user_count,
            "degree_per_graph": degree,
            "seeds": seeds,
            "target_coverage": target_coverage,
            "min_training_batches": min_batches,
            "max_coverage_batches": max_batches,
        },
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
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261116])
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--min-batches", type=int, default=620)
    parser.add_argument("--max-batches", type=int, default=10_000)
    args = parser.parse_args()
    try:
        payload = run_case(
            user_count=args.users,
            degree=args.degree,
            seeds=args.seeds,
            target_coverage=args.target_coverage,
            min_batches=args.min_batches,
            max_batches=args.max_batches,
        )
    except CoverageError as error:
        payload = {"status": "failed", "error": str(error)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

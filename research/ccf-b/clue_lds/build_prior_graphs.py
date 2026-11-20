from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np


Edge = tuple[int, int]


def normalize_edge(left: int, right: int) -> Edge:
    if left == right:
        raise ValueError("self-loops are not allowed")
    return (left, right) if left < right else (right, left)


def top_m_edges(prior: np.ndarray, top_m: int) -> set[Edge]:
    edges: set[Edge] = set()
    for user in range(prior.shape[0]):
        order = np.argsort(-prior[user])
        chosen = [int(candidate) for candidate in order if candidate != user][
            :top_m
        ]
        edges.update(normalize_edge(user, candidate) for candidate in chosen)
    return edges


def permuted_prior(prior: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(prior.shape[0])
    permuted = prior[np.ix_(permutation, permutation)]
    return permuted.astype(np.float32), permutation


def degree_preserving_random(
    edges: set[Edge],
    user_count: int,
    seed: int,
    max_attempts: int = 100,
) -> set[Edge]:
    degrees = np.zeros(user_count, dtype=np.int64)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    rng = random.Random(seed)
    stubs = [
        user for user, degree in enumerate(degrees) for _ in range(int(degree))
    ]
    for _ in range(max_attempts):
        candidate_stubs = stubs.copy()
        rng.shuffle(candidate_stubs)
        candidate: set[Edge] = set()
        valid = True
        while candidate_stubs:
            left = candidate_stubs.pop()
            if not candidate_stubs:
                valid = False
                break
            chosen_index = None
            for index, right in enumerate(candidate_stubs):
                if right != left and normalize_edge(left, right) not in candidate:
                    chosen_index = index
                    break
            if chosen_index is None:
                valid = False
                break
            right = candidate_stubs.pop(chosen_index)
            candidate.add(normalize_edge(left, right))
        if valid and candidate != edges:
            return candidate
    raise RuntimeError("could not build a distinct degree-preserving graph")


def edge_array(edges: set[Edge]) -> np.ndarray:
    return np.asarray(sorted(edges), dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()

    prior = np.load(args.prior)
    base_edges = top_m_edges(prior, args.top_m)
    permuted, permutation = permuted_prior(prior, args.seed + 1)
    permuted_edges = top_m_edges(permuted, args.top_m)
    random_edges = degree_preserving_random(
        base_edges, prior.shape[0], args.seed + 2
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "clue_prior_graphs.npz",
        A3=edge_array(base_edges),
        A5a=edge_array(random_edges),
        A5b=edge_array(permuted_edges),
        permutation=permutation,
        prior=prior,
    )
    metadata = {
        "top_m": args.top_m,
        "seed": args.seed,
        "user_count": int(prior.shape[0]),
        "edges": {
            "A3": len(base_edges),
            "A5a": len(random_edges),
            "A5b": len(permuted_edges),
        },
        "degree_sequence_preserved": True,
        "A5b_uses_permuted_profile_prior": True,
    }
    (args.output_dir / "clue_prior_graphs_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

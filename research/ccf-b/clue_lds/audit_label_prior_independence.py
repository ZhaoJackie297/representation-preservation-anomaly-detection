from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rank_average_ties(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_ranks = rank_average_ties(left)
    right_ranks = rank_average_ties(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 0:
        raise ValueError("Spearman correlation is undefined for constant data")
    return float(left_centered @ right_centered / denominator)


def unique_label_pairs(
    rows: Iterable[dict[str, str]],
) -> list[tuple[str, str, float]]:
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in rows:
        source = row["source_uid"]
        replacement = row["replacement_uid"]
        if source == replacement:
            raise ValueError("self-pairs are not allowed")
        key = tuple(sorted((source, replacement)))
        grouped.setdefault(key, []).append((source, float(row["similarity"])))

    pairs: list[tuple[str, str, float]] = []
    for (left, right), entries in sorted(grouped.items()):
        if len(entries) != 2:
            raise ValueError(
                f"pair {left}<->{right} has {len(entries)} label rows, expected 2"
            )
        directions = {source for source, _ in entries}
        if directions != {left, right}:
            raise ValueError(
                f"pair {left}<->{right} does not contain both directions"
            )
        similarities = [similarity for _, similarity in entries]
        if not np.isclose(similarities[0], similarities[1], rtol=0, atol=1e-12):
            raise ValueError(
                f"pair {left}<->{right} has inconsistent similarity values"
            )
        pairs.append((left, right, similarities[0]))
    return pairs


def audit_label_prior_independence(
    labels_path: Path,
    profile_distances_path: Path,
    profile_metadata_path: Path,
    *,
    maximum_abs_rho: float = 0.5,
) -> dict[str, object]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = unique_label_pairs(rows)

    metadata = json.loads(profile_metadata_path.read_text(encoding="utf-8"))
    users = [str(user) for user in metadata["users_order"]]
    user_index = {user: index for index, user in enumerate(users)}
    distances = np.load(profile_distances_path)
    if distances.shape != (len(users), len(users)):
        raise ValueError(
            "profile distance shape does not match users_order: "
            f"{distances.shape} vs {len(users)}"
        )
    if not np.allclose(distances, distances.T, atol=1e-6):
        raise ValueError("profile distance matrix is not symmetric")

    missing = sorted(
        {
            user
            for left, right, _ in pairs
            for user in (left, right)
            if user not in user_index
        }
    )
    if missing:
        raise ValueError(f"label users missing from profile metadata: {missing}")

    label_similarities = np.asarray(
        [similarity for _, _, similarity in pairs], dtype=np.float64
    )
    profile_distances = np.asarray(
        [
            distances[user_index[left], user_index[right]]
            for left, right, _ in pairs
        ],
        dtype=np.float64,
    )
    rho_similarity_distance = spearman(
        label_similarities, profile_distances
    )
    rho_similarity_affinity = spearman(
        label_similarities, -profile_distances
    )
    passed = abs(rho_similarity_affinity) < maximum_abs_rho
    return {
        "labels": str(labels_path),
        "labels_sha256": file_sha256(labels_path),
        "profile_distances": str(profile_distances_path),
        "profile_distances_sha256": file_sha256(profile_distances_path),
        "profile_metadata_sha256": file_sha256(profile_metadata_path),
        "event_rows": len(rows),
        "unique_pairs": len(pairs),
        "label_similarity_min": float(label_similarities.min()),
        "label_similarity_max": float(label_similarities.max()),
        "profile_distance_min": float(profile_distances.min()),
        "profile_distance_median": float(np.median(profile_distances)),
        "profile_distance_max": float(profile_distances.max()),
        "spearman_label_similarity_vs_profile_distance": rho_similarity_distance,
        "spearman_label_similarity_vs_profile_affinity": rho_similarity_affinity,
        "maximum_abs_rho": maximum_abs_rho,
        "status": (
            "PASS_DISTANCES_DISTINCT"
            if passed
            else "FAIL_REQUIRES_INDEPENDENT_DISTANCE_CONTROL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--profile-distances", type=Path, required=True)
    parser.add_argument("--profile-metadata", type=Path, required=True)
    parser.add_argument("--maximum-abs-rho", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_label_prior_independence(
        args.labels,
        args.profile_distances,
        args.profile_metadata,
        maximum_abs_rho=args.maximum_abs_rho,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if result["status"] != "PASS_DISTANCES_DISTINCT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

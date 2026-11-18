from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import math
import random
from typing import Iterable, Mapping


Edge = tuple[int, int]


class CoverageError(RuntimeError):
    """Raised when a coverage table cannot satisfy the preregistered gates."""


def normalize_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("self-loops are not valid user relation edges")
    return (u, v) if u < v else (v, u)


def normalize_edge_set(edges: Iterable[Edge], user_count: int) -> frozenset[Edge]:
    normalized: set[Edge] = set()
    for u, v in edges:
        if not (0 <= u < user_count and 0 <= v < user_count):
            raise ValueError(f"edge {(u, v)} is outside [0, {user_count})")
        normalized.add(normalize_edge(int(u), int(v)))
    return frozenset(normalized)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _edge_priority(seed: int, stage: int, edge: Edge) -> bytes:
    payload = f"{seed}|{stage}|{edge[0]}|{edge[1]}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _batch_pairs(batch: Iterable[int]) -> set[Edge]:
    return {normalize_edge(u, v) for u, v in combinations(sorted(batch), 2)}


@dataclass(frozen=True)
class CoverageResult:
    user_count: int
    seed: int
    stage: int
    batch_size: int
    designed_slots: int
    filler_slots: int
    target_coverage: float
    coverage_batches: tuple[tuple[int, ...], ...]
    designed_batches: tuple[tuple[int, ...], ...]
    filler_batches: tuple[tuple[int, ...], ...]
    training_batches: tuple[tuple[int, ...], ...]
    intended_edges: Mapping[str, int]
    realized_edges: Mapping[str, int]
    coverage: Mapping[str, float]
    edge_exposures: Mapping[str, Mapping[Edge, int]]
    user_exposures: Mapping[int, int]
    batch_table_sha256: str
    training_batch_table_sha256: str
    training_edge_exposures: Mapping[str, Mapping[Edge, int]]

    def to_summary(self) -> dict[str, object]:
        return {
            "user_count": self.user_count,
            "seed": self.seed,
            "stage": self.stage,
            "batch_size": self.batch_size,
            "designed_slots": self.designed_slots,
            "filler_slots": self.filler_slots,
            "target_coverage": self.target_coverage,
            "coverage_batch_count": len(self.coverage_batches),
            "training_batch_count": len(self.training_batches),
            "intended_edges": dict(self.intended_edges),
            "realized_edges": dict(self.realized_edges),
            "coverage": dict(self.coverage),
            "users_with_zero_exposure": sum(
                value == 0 for value in self.user_exposures.values()
            ),
            "median_user_exposure": _median(self.user_exposures.values()),
            "min_user_exposure": min(self.user_exposures.values(), default=0),
            "batch_table_sha256": self.batch_table_sha256,
            "training_batch_table_sha256": self.training_batch_table_sha256,
            "median_training_edge_exposure": {
                name: _median(values.values())
                for name, values in self.training_edge_exposures.items()
            },
        }


def _median(values: Iterable[int]) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _hash_batches(batches: Iterable[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(",".join(str(index) for index in batch).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def _compute_exposures(
    batches: Iterable[tuple[int, ...]],
    target_sets: Mapping[str, frozenset[Edge]],
    user_count: int,
) -> tuple[
    dict[str, dict[Edge, int]],
    dict[int, int],
]:
    edge_exposures = {
        name: {edge: 0 for edge in edges} for name, edges in target_sets.items()
    }
    user_exposures = {user: 0 for user in range(user_count)}
    for batch in batches:
        for edge in _batch_pairs(batch):
            touched = False
            for name, edges in target_sets.items():
                if edge in edges:
                    edge_exposures[name][edge] += 1
                    touched = True
            if touched:
                user_exposures[edge[0]] += 1
                user_exposures[edge[1]] += 1
    return edge_exposures, user_exposures


def build_common_coverage_table(
    *,
    user_count: int,
    target_edge_sets: Mapping[str, Iterable[Edge]],
    seed: int,
    stage: int,
    batch_size: int = 16,
    designed_slots: int = 8,
    target_coverage: float = 0.8,
    min_training_batches: int = 0,
    max_coverage_batches: int = 10_000,
) -> CoverageResult:
    """Build a seed-dependent table with fixed designed and random filler slots.

    The coverage target is checked independently for every named graph. The
    table succeeds only if every user incident to at least one target edge is
    exposed at least once.
    """

    if user_count < batch_size:
        raise ValueError("user_count must be at least batch_size")
    if not 0 < designed_slots < batch_size:
        raise ValueError("designed_slots must be in (0, batch_size)")
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    if max_coverage_batches <= 0:
        raise ValueError("max_coverage_batches must be positive")

    filler_slots = batch_size - designed_slots
    target_sets = {
        name: normalize_edge_set(edges, user_count)
        for name, edges in target_edge_sets.items()
    }
    if not target_sets or any(not edges for edges in target_sets.values()):
        raise ValueError("every target graph must contain at least one edge")

    intended = {name: len(edges) for name, edges in target_sets.items()}
    required = {
        name: math.ceil(target_coverage * count)
        for name, count in intended.items()
    }

    edge_memberships: dict[Edge, set[str]] = {}
    union_edges: set[Edge] = set()
    adjacency: dict[int, set[int]] = {user: set() for user in range(user_count)}
    for name, edges in target_sets.items():
        for edge in edges:
            union_edges.add(edge)
            edge_memberships.setdefault(edge, set()).add(name)
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
    edge_priorities = {
        edge: _edge_priority(seed, stage, edge) for edge in union_edges
    }

    users_with_edges = {
        user
        for edge in union_edges
        for user in edge
    }
    uncovered = set(union_edges)
    realized_by_graph = {name: 0 for name in target_sets}
    realized_edges = {name: set() for name in target_sets}
    user_seen = {user: False for user in range(user_count)}
    rng = random.Random(_stable_seed("coverage", seed, stage))

    coverage_batches: list[tuple[int, ...]] = []
    designed_batches: list[tuple[int, ...]] = []
    filler_batches: list[tuple[int, ...]] = []

    def coverage(name: str) -> float:
        return realized_by_graph[name] / intended[name]

    def gates_met() -> bool:
        return (
            all(coverage(name) >= target_coverage for name in target_sets)
            and all(user_seen[user] for user in users_with_edges)
        )

    while not gates_met():
        if len(coverage_batches) >= max_coverage_batches:
            summary = {
                name: {
                    "coverage": coverage(name),
                    "intended": intended[name],
                    "realized": realized_by_graph[name],
                }
                for name in target_sets
            }
            missing_users = [
                user for user in users_with_edges if not user_seen[user]
            ]
            raise CoverageError(
                "coverage budget exhausted: "
                f"graphs={summary}, missing_users={len(missing_users)}"
            )

        zero_user_edges = [
            edge
            for edge in uncovered
            if not user_seen[edge[0]] or not user_seen[edge[1]]
        ]
        pool = zero_user_edges if zero_user_edges else list(uncovered)
        if not pool:
            # A graph can already meet its edge target while an isolated user
            # remains unseen. Restart from all edges touching such users.
            pool = [
                edge
                for edge in union_edges
                if not user_seen[edge[0]] or not user_seen[edge[1]]
            ]
        start_edge = min(
            pool,
            key=lambda edge: (
                max(
                    target_coverage - coverage(name)
                    for name in edge_memberships[edge]
                )
                * -1.0,
                edge_priorities[edge],
            ),
        )

        designed: list[int] = []
        for user in start_edge:
            if user not in designed:
                designed.append(user)

        while len(designed) < designed_slots:
            designed_set = set(designed)
            candidates = {
                neighbor
                for user in designed
                for neighbor in adjacency[user]
                if neighbor not in designed_set
            }
            if not candidates:
                candidates = {
                    user
                    for user in range(user_count)
                    if user not in designed_set
                }

            def candidate_key(user: int) -> tuple[int, int, bytes]:
                gain = sum(
                    normalize_edge(user, anchor) in uncovered
                    for anchor in designed
                )
                candidate_edge = normalize_edge(user, designed[0])
                priority = edge_priorities.get(candidate_edge)
                if priority is None:
                    priority = _edge_priority(seed, stage, candidate_edge)
                return (
                    gain,
                    0 if not user_seen[user] else 1,
                    priority,
                )

            chosen = max(candidates, key=candidate_key)
            designed.append(chosen)

        unavailable = set(designed)
        filler_pool = [
            user for user in range(user_count) if user not in unavailable
        ]
        fillers = rng.sample(filler_pool, filler_slots)
        batch = tuple(sorted(designed + fillers))

        for edge in _batch_pairs(batch):
            memberships = edge_memberships.get(edge)
            if not memberships:
                continue
            user_seen[edge[0]] = True
            user_seen[edge[1]] = True
            uncovered.discard(edge)
            for name in memberships:
                if edge not in realized_edges[name]:
                    realized_edges[name].add(edge)
                    realized_by_graph[name] += 1

        coverage_batches.append(batch)
        designed_batches.append(tuple(sorted(designed)))
        filler_batches.append(tuple(sorted(fillers)))

    coverage_values = {
        name: realized_by_graph[name] / intended[name] for name in target_sets
    }

    if len(coverage_batches) < min_training_batches:
        repeats = math.ceil(min_training_batches / len(coverage_batches))
        training_batches = tuple(
            (coverage_batches * repeats)[:min_training_batches]
        )
    else:
        training_batches = tuple(coverage_batches)

    edge_exposures, user_exposures = _compute_exposures(
        coverage_batches, target_sets, user_count
    )
    training_edge_exposures, _ = _compute_exposures(
        training_batches, target_sets, user_count
    )

    return CoverageResult(
        user_count=user_count,
        seed=seed,
        stage=stage,
        batch_size=batch_size,
        designed_slots=designed_slots,
        filler_slots=filler_slots,
        target_coverage=target_coverage,
        coverage_batches=tuple(coverage_batches),
        designed_batches=tuple(designed_batches),
        filler_batches=tuple(filler_batches),
        training_batches=training_batches,
        intended_edges=intended,
        realized_edges={
            name: len(edges) for name, edges in realized_edges.items()
        },
        coverage=coverage_values,
        edge_exposures=edge_exposures,
        user_exposures=user_exposures,
        batch_table_sha256=_hash_batches(tuple(coverage_batches)),
        training_batch_table_sha256=_hash_batches(training_batches),
        training_edge_exposures=training_edge_exposures,
    )


def pair_jaccard(left: CoverageResult, right: CoverageResult) -> float:
    left_pairs: set[Edge] = set()
    right_pairs: set[Edge] = set()
    for batch in left.coverage_batches:
        left_pairs.update(_batch_pairs(batch))
    for batch in right.coverage_batches:
        right_pairs.update(_batch_pairs(batch))
    union = left_pairs | right_pairs
    if not union:
        return 1.0
    return len(left_pairs & right_pairs) / len(union)

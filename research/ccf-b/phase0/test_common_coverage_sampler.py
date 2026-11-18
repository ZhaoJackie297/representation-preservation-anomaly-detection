from __future__ import annotations

import unittest

from common_coverage_sampler import (
    build_common_coverage_table,
    normalize_edge,
    normalize_edge_set,
    pair_jaccard,
)


def ring_graph(user_count: int, degree: int) -> set[tuple[int, int]]:
    half = max(1, degree // 2)
    return {
        normalize_edge(user, (user + offset) % user_count)
        for user in range(user_count)
        for offset in range(1, half + 1)
    }


class CommonCoverageSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.user_count = 64
        self.targets = {
            "A3": ring_graph(self.user_count, 8),
            "A5a": {
                normalize_edge(user, (user * 17 + 11) % self.user_count)
                for user in range(self.user_count)
            },
            "A5b": {
                normalize_edge(user, (user * 23 + 7) % self.user_count)
                for user in range(self.user_count)
            },
        }

    def build(self, seed: int):
        return build_common_coverage_table(
            user_count=self.user_count,
            target_edge_sets=self.targets,
            seed=seed,
            stage=0,
            target_coverage=0.8,
            min_training_batches=32,
            max_coverage_batches=1_000,
        )

    def test_reproducible_for_same_seed(self) -> None:
        left = self.build(17)
        right = self.build(17)
        self.assertEqual(left.training_batches, right.training_batches)
        self.assertEqual(left.batch_table_sha256, right.batch_table_sha256)

    def test_seed_changes_table(self) -> None:
        left = self.build(17)
        right = self.build(19)
        self.assertNotEqual(left.batch_table_sha256, right.batch_table_sha256)
        self.assertLess(pair_jaccard(left, right), 1.0)

    def test_ten_official_seed_hashes_are_unique(self) -> None:
        seeds = list(range(20261116, 20261126))
        hashes = {self.build(seed).batch_table_sha256 for seed in seeds}
        self.assertEqual(len(hashes), len(seeds))

    def test_exact_half_random_fill(self) -> None:
        result = self.build(17)
        self.assertEqual(result.designed_slots, 8)
        self.assertEqual(result.filler_slots, 8)
        for designed, filler in zip(
            result.designed_batches, result.filler_batches
        ):
            self.assertEqual(len(designed), 8)
            self.assertEqual(len(filler), 8)
            self.assertFalse(set(designed) & set(filler))

    def test_graph_specific_coverage_gates(self) -> None:
        result = self.build(17)
        for graph in self.targets:
            self.assertGreaterEqual(result.coverage[graph], 0.8)
        self.assertEqual(
            sum(count == 0 for count in result.user_exposures.values()), 0
        )

    def test_train_table_reaches_minimum_length(self) -> None:
        result = self.build(17)
        self.assertGreaterEqual(len(result.training_batches), 32)

    def test_duplicate_edges_are_normalized(self) -> None:
        edges = normalize_edge_set(
            [(0, 1), (1, 0), (0, 1)], user_count=4
        )
        self.assertEqual(edges, frozenset({(0, 1)}))


if __name__ == "__main__":
    unittest.main()

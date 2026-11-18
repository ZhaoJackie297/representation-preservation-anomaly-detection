from __future__ import annotations

import unittest

import numpy as np

from build_prior_graphs import (
    degree_preserving_random,
    normalize_edge,
    top_m_edges,
)
from build_profile_prior import gaussian_prior, profile_matrix


class ProfileGraphTest(unittest.TestCase):
    def test_profile_and_prior(self) -> None:
        values = np.asarray(
            [
                [[1.0, 2.0], [2.0, 3.0], [np.nan, np.nan]],
                [[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
            ]
        )
        active = np.asarray(
            [[True, True, False], [True, True, True], [True, True, True]]
        )
        profiles, fractions = profile_matrix(values, active, train_end=1)
        prior, distances = gaussian_prior(profiles)
        self.assertEqual(profiles.shape, (3, 5))
        self.assertEqual(prior.shape, (3, 3))
        self.assertAlmostEqual(float(np.sum(prior[0])), 1.0)
        self.assertEqual(distances.shape, (3, 3))
        self.assertEqual(len(fractions), 3)

    def test_top_m_and_random_graph(self) -> None:
        prior = np.asarray(
            [
                [0.0, 0.8, 0.1, 0.1],
                [0.8, 0.0, 0.1, 0.1],
                [0.1, 0.1, 0.0, 0.8],
                [0.1, 0.1, 0.8, 0.0],
            ]
        )
        edges = top_m_edges(prior, 1)
        self.assertEqual(
            edges,
            {
                normalize_edge(0, 1),
                normalize_edge(2, 3),
            },
        )
        random_edges = degree_preserving_random(edges, 4, 1)
        self.assertNotEqual(random_edges, edges)
        degree = np.zeros(4, dtype=int)
        for left, right in random_edges:
            degree[left] += 1
            degree[right] += 1
        self.assertEqual(degree.tolist(), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

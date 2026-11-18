from __future__ import annotations

import unittest

import numpy as np

from audit_cert_profile_information import (
    gaussian_prior,
    knn_bandwidth,
    pairwise_distances,
)


class CertProfileInformationTest(unittest.TestCase):
    def test_pairwise_distances_are_symmetric_and_zero_diagonal(self) -> None:
        features = np.asarray([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        distances = pairwise_distances(features)
        np.testing.assert_allclose(distances, distances.T)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        self.assertAlmostEqual(distances[0, 1], 5.0)

    def test_knn_bandwidth_uses_off_diagonal_distance(self) -> None:
        distances = np.asarray(
            [
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 4.0],
                [5.0, 4.0, 0.0],
            ]
        )
        self.assertEqual(knn_bandwidth(distances, k=1), 1.0)

    def test_gaussian_prior_rows_are_normalized(self) -> None:
        distances = np.asarray(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        )
        prior = gaussian_prior(distances, bandwidth=1.0)
        np.testing.assert_allclose(prior.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.diag(prior), 0.0)


if __name__ == "__main__":
    unittest.main()

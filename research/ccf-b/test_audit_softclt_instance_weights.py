from __future__ import annotations

import unittest

import numpy as np

from audit_softclt_instance_weights import (
    make_soft,
    normalized_rows,
    toy_losses,
    weight_statistics,
)


class SoftCLTWeightAuditTest(unittest.TestCase):
    def test_standard_soft_matrix_has_more_than_one_unit_mass(self) -> None:
        stats = weight_statistics(make_soft(4, diagonal=1.0, off_diagonal=0.5))
        self.assertAlmostEqual(stats["row_sum_mean"], 2.5)
        self.assertGreater(stats["off_diagonal_fraction"], 0.0)

    def test_row_normalization_changes_sum(self) -> None:
        soft = make_soft(4)
        normalized = normalized_rows(soft)
        np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
        self.assertFalse(np.allclose(normalized, soft))

    def test_official_loss_scales_with_weight_scale(self) -> None:
        soft = make_soft(4)
        losses = toy_losses(soft)
        self.assertAlmostEqual(losses["loss_scaled_2x"], 2.0 * losses["loss_raw"])


if __name__ == "__main__":
    unittest.main()

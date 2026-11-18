from __future__ import annotations

import unittest

import numpy as np

from risk_control_check import (
    cusum,
    cusum_mean_delay,
    ewma,
    ewma_mean_delay,
    simulate_method,
)


class RiskControlCheckTest(unittest.TestCase):
    def test_ewma_and_cusum_are_deterministic(self) -> None:
        values = np.asarray([0.0, 1.0, 2.0, 0.0])
        np.testing.assert_allclose(ewma(values, 0.5), [0.0, 0.5, 1.25, 0.625])
        np.testing.assert_allclose(cusum(values, 0.5), [0.0, 0.5, 2.0, 1.5])

    def test_delay_approximations_are_positive(self) -> None:
        self.assertGreater(ewma_mean_delay(0.2, 1.0, 2.0), 0.0)
        self.assertGreater(cusum_mean_delay(0.5, 2.0, 2.0), 0.0)

    def test_simulation_reports_finite_delay(self) -> None:
        result = simulate_method(
            method="ewma",
            parameter=0.2,
            threshold=0.5,
            shift=2.0,
            repetitions=100,
            horizon=40,
            seed=3,
            change_index=10,
        )
        self.assertGreater(result["mean_detection_delay"], 0.0)


if __name__ == "__main__":
    unittest.main()

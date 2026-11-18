from __future__ import annotations

import unittest

from power_cert_gate import binomial_event_mde, projected_mde


class CertGatePowerTest(unittest.TestCase):
    def test_projected_mde_uses_ten_seed_target(self) -> None:
        result = projected_mde([0.01, 0.02, 0.03])
        self.assertEqual(result["projected_seeds"], 10)
        self.assertGreater(result["projected_se_mde"], 0.0)

    def test_binomial_mde_decreases_with_correlation(self) -> None:
        independent = binomial_event_mde(0.2, 99, correlation=0.0)
        correlated = binomial_event_mde(0.2, 99, correlation=0.8)
        self.assertGreater(independent, correlated)


if __name__ == "__main__":
    unittest.main()

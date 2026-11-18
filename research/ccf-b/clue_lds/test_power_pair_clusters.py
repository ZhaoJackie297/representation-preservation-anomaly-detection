from __future__ import annotations

import unittest

from power_pair_clusters import find_mde, power_for_shift


class PairClusterPowerTest(unittest.TestCase):
    def test_power_increases_with_effect(self) -> None:
        small, _ = power_for_shift(
            baseline_recall=0.25,
            log_odds_shift=0.0,
            cluster_sd=1.0,
            pairs=20,
            events_per_pair=2,
            repetitions=300,
            seed=1,
        )
        large, _ = power_for_shift(
            baseline_recall=0.25,
            log_odds_shift=3.0,
            cluster_sd=1.0,
            pairs=20,
            events_per_pair=2,
            repetitions=300,
            seed=1,
        )
        self.assertLess(small, large)

    def test_mde_is_positive(self) -> None:
        result = find_mde(
            baseline_recall=0.25,
            cluster_sd=1.0,
            pairs=20,
            events_per_pair=2,
            repetitions=300,
            target_power=0.8,
            seed=1,
        )
        self.assertGreater(result["absolute_recall_delta"], 0)


if __name__ == "__main__":
    unittest.main()

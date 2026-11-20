from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime" / "cert"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from feature_contracts import (  # noqa: E402
    build_user_profiles,
    sanitize_user_day_features,
)
from production_protocol_r42 import build_alerts  # noqa: E402


class FeatureContractTest(unittest.TestCase):
    def test_release_sample_keeps_contract_columns(self) -> None:
        sample = pd.read_csv(
            REPO / "data" / "samples" / "r4.2_user_day_features_sample.csv"
        )
        self.assertGreater(len(sample), 0)
        self.assertTrue(
            {
                "user_id",
                "date",
                "logon_count",
                "first_logon_minute",
                "last_logoff_minute",
                "active_span_minutes",
            }.issubset(sample.columns)
        )
        self.assertFalse(sample.duplicated(["user_id", "date"]).any())

    def test_timing_sentinel_stays_missing(self) -> None:
        values = np.asarray(
            [[[2.0, -1.0], [0.0, 720.0]]],
            dtype=np.float64,
        )
        output = sanitize_user_day_features(values, count_dimensions=1)
        self.assertAlmostEqual(float(output[0, 0, 0]), np.log1p(2.0))
        self.assertTrue(np.isnan(output[0, 0, 1]))
        self.assertEqual(float(output[0, 1, 1]), 720.0)

    def test_profiles_ignore_validation_and_test_values(self) -> None:
        raw = np.asarray(
            [
                [[1.0, 2.0], [2.0, 2.5], [3.0, 3.0], [4.0, 4.0]],
                [[4.0, 5.0], [5.0, 5.5], [6.0, 6.0], [7.0, 7.0]],
            ],
            dtype=np.float64,
        )
        active = np.ones((2, 4), dtype=bool)
        expected = build_user_profiles(raw, active, train_end=1)
        changed = raw.copy()
        changed[:, 2:] = 1_000_000.0
        actual = build_user_profiles(changed, active, train_end=1)
        np.testing.assert_allclose(actual, expected)

    def test_top_k_and_cooldown_are_identical(self) -> None:
        scores = np.asarray(
            [
                [9.0, 8.0, 7.0, 6.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        active = np.ones_like(scores, dtype=bool)
        active[2, 1] = False
        scores[2, 1] = 100.0
        alerts = build_alerts(scores, active, k=1, cooldown_days=3)
        self.assertEqual(alerts[0].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(alerts[1].tolist(), [0, 1, 0, 0, 1])
        self.assertEqual(alerts[2].tolist(), [0, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()

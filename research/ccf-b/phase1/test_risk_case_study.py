from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk_case_study import (
    causal_cusum,
    causal_ewma,
    incident_details,
    neighbor_score,
    propagate_scores,
    top_m_neighbors,
)


class RiskCaseStudyTest(unittest.TestCase):
    def test_ewma_is_causal_and_resets(self) -> None:
        score = np.asarray([[1.0, 2.0, np.nan, 3.0]])
        active = np.asarray([[True, True, False, True]])
        output = causal_ewma(score, active, 0.5)
        self.assertEqual(output[0, 0], 0.5)
        self.assertEqual(output[0, 1], 1.25)
        self.assertTrue(np.isnan(output[0, 2]))
        self.assertEqual(output[0, 3], 1.5)

    def test_cusum_is_nonnegative(self) -> None:
        score = np.asarray([[0.0, 0.2, 2.0, 0.0]])
        active = np.ones_like(score, dtype=bool)
        output = causal_cusum(score, active, 0.5)
        self.assertTrue(np.all(output >= 0.0))
        self.assertGreater(output[0, 2], output[0, 1])

    def test_top_m_graph_is_row_normalized(self) -> None:
        graph = np.asarray(
            [
                [0.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, 2.0, 4.0],
                [5.0, 1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0, 0.0],
            ]
        )
        normalized = top_m_neighbors(graph, 2)
        np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
        self.assertEqual(normalized[0, 1], 0.6)
        self.assertEqual(normalized[0, 2], 0.4)

    def test_propagation_uses_active_neighbors(self) -> None:
        scores = np.asarray([[1.0], [3.0], [100.0]])
        active = np.asarray([[True], [True], [False]])
        graph = np.asarray(
            [
                [0.0, 0.5, 0.5],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        neighbor = neighbor_score(scores, active, graph)
        self.assertEqual(neighbor[0, 0], 3.0)
        propagated = propagate_scores(scores, active, graph, 0.2)
        self.assertAlmostEqual(propagated[0, 0], 1.4)

    def test_incident_details_respects_horizon(self) -> None:
        alerts = np.zeros((1, 5), dtype=np.uint8)
        alerts[0, 3] = 1
        incidents = pd.DataFrame(
            [
                {
                    "incident_id": "x",
                    "scenario": 1,
                    "observed_accounts": "u",
                    "start": "2026-01-01",
                    "end": "2026-01-05",
                }
            ]
        )
        details = incident_details(
            alerts,
            np.asarray(["u"]),
            pd.date_range("2026-01-01", periods=5),
            incidents,
            0,
            4,
            3,
        )
        self.assertFalse(bool(details.iloc[0]["hit"]))


if __name__ == "__main__":
    unittest.main()

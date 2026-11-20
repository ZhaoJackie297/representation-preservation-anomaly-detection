from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analyze_cert_paired_pilot import (
    load_baseline_metrics,
    load_pipeline_metrics,
    paired_rows,
)


class AnalyzeCERTPairedPilotTest(unittest.TestCase):
    def test_pipeline_table_collapses_k_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            rows = []
            for seed in (20261201, 20261202):
                for k, recall in ((5, 0.1), (10, 0.2), (20, 0.3)):
                    rows.append(
                        {
                            "seed": seed,
                            "method": "T4_full_soft_global_z",
                            "split": "test",
                            "k": k,
                            "pr_auc": 0.01,
                            "roc_auc": 0.8,
                            "event_recall_3d": recall,
                        }
                    )
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = load_pipeline_metrics(
                path,
                "T4_full_soft_global_z",
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded["event_recall_k10"].tolist(), [0.2, 0.2])

    def test_paired_delta_uses_matching_seed(self) -> None:
        left = pd.DataFrame(
            {
                "seed": [1, 2],
                "pr_auc": [0.3, 0.4],
                "roc_auc": [0.8, 0.9],
                "event_recall_k5": [0.2, 0.3],
                "event_recall_k10": [0.4, 0.5],
                "event_recall_k20": [0.6, 0.7],
            }
        )
        right = left.copy()
        right["pr_auc"] = [0.1, 0.2]
        output = paired_rows(left, right, comparison="left_minus_right")
        pr = output.loc[output["metric"] == "pr_auc"]
        self.assertTrue(
            all(abs(value - 0.2) < 1e-12 for value in pr["delta"])
        )

    def test_baseline_table_requires_one_row_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.csv"
            pd.DataFrame(
                [
                    {
                        "method": "t-NNCLR",
                        "seed": 20261201,
                        "pr_auc": 0.01,
                        "roc_auc": 0.7,
                        "event_recall_k5": 0.1,
                        "event_recall_k10": 0.2,
                        "event_recall_k20": 0.3,
                    }
                ]
            ).to_csv(path, index=False)
            loaded = load_baseline_metrics(path, "t-NNCLR")
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()

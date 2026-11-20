from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from summarize_smd_baselines import load_seed


class SummarizeSMDBaselinesTest(unittest.TestCase):
    def test_load_seed_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_root = root / "seed_20261201"
            seed_root.mkdir()
            pd.DataFrame(
                {
                    "machine": ["m1", "m2"],
                    "pai_pr_auc": [0.1, 0.3],
                    "pai_roc_auc": [0.5, 0.7],
                    "pai_pa_f1_top5": [0.2, 0.4],
                }
            ).to_csv(seed_root / "results_by_machine.csv", index=False)
            row = load_seed(
                root,
                method="test",
                seed=20261201,
                prefix="pai_",
            )
            self.assertAlmostEqual(row["pr_auc"], 0.2)
            self.assertEqual(row["machines"], 2)


if __name__ == "__main__":
    unittest.main()

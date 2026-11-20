from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from compare_smd_deep_baselines import load


class CompareSMDDeepBaselinesTest(unittest.TestCase):
    def test_load_seed_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "seed_20261201"
            directory.mkdir()
            pd.DataFrame(
                {
                    "machine": ["m2", "m1"],
                    "pai_pr_auc": [0.2, 0.1],
                }
            ).to_csv(directory / "results_by_machine.csv", index=False)
            frame = load(root, 20261201)
            self.assertEqual(frame["machine"].tolist(), ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()

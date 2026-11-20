from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analyze_smd_multiseed import METRICS, summarize_pair


class AnalyzeSMDMultiseedTest(unittest.TestCase):
    def test_known_pair_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            t0_root = root / "t0"
            t4_root = root / "t4"
            for seed in (20261201, 20261202):
                for variant_root, offset in ((t0_root, 0.0), (t4_root, 0.1)):
                    directory = variant_root / f"seed_{seed}"
                    directory.mkdir(parents=True)
                    rows = []
                    for index, machine in enumerate(("m1", "m2")):
                        row = {"machine": machine, "seed": seed}
                        for variant in ("original", "cos", "euc", "pai"):
                            for metric in METRICS:
                                row[f"{variant}_{metric}"] = (
                                    0.5 + index * 0.1 + offset
                                )
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(
                        directory / "results_by_machine.csv",
                        index=False,
                    )
            seed_rows, bootstrap_rows, summary = summarize_pair(
                t0_root,
                t4_root,
                [20261201, 20261202],
                name="pai_t4_vs_t0",
                variant="pai",
                bootstrap_iterations=100,
            )
            self.assertEqual(len(seed_rows), 2 * len(METRICS))
            self.assertEqual(len(bootstrap_rows), len(METRICS))
            self.assertAlmostEqual(summary["pr_auc"]["seed_mean_delta"], 0.1)
            self.assertEqual(summary["pr_auc"]["positive_seeds"], 2)


if __name__ == "__main__":
    unittest.main()

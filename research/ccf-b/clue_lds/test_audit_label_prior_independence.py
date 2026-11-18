from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_label_prior_independence import (
    audit_label_prior_independence,
    spearman,
    unique_label_pairs,
)


class AuditLabelPriorIndependenceTest(unittest.TestCase):
    def test_spearman_detects_positive_and_negative_order(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 5, 7]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [7, 5, 3]), -1.0)

    def test_rejects_inconsistent_pair_similarity(self) -> None:
        rows = [
            {"source_uid": "u1", "replacement_uid": "u2", "similarity": "0.4"},
            {"source_uid": "u2", "replacement_uid": "u1", "similarity": "0.5"},
        ]
        with self.assertRaises(ValueError):
            unique_label_pairs(rows)

    def test_end_to_end_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source_uid", "replacement_uid", "similarity"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "source_uid": "u1",
                            "replacement_uid": "u2",
                            "similarity": "0.2",
                        },
                        {
                            "source_uid": "u2",
                            "replacement_uid": "u1",
                            "similarity": "0.2",
                        },
                        {
                            "source_uid": "u3",
                            "replacement_uid": "u4",
                            "similarity": "0.4",
                        },
                        {
                            "source_uid": "u4",
                            "replacement_uid": "u3",
                            "similarity": "0.4",
                        },
                    ]
                )
            distances = root / "distances.npy"
            np.save(
                distances,
                np.asarray(
                    [
                        [0.0, 3.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                ),
            )
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps({"users_order": ["u1", "u2", "u3", "u4"]}),
                encoding="utf-8",
            )

            result = audit_label_prior_independence(
                labels, distances, metadata
            )

            self.assertEqual(result["unique_pairs"], 2)
            self.assertGreater(
                result["spearman_label_similarity_vs_profile_affinity"], 0.0
            )


if __name__ == "__main__":
    unittest.main()

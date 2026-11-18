from __future__ import annotations

import unittest

import numpy as np

from audit_three_channel_datasets import aggregate_batches


class ThreeChannelDatasetAuditTest(unittest.TestCase):
    def test_batch_aggregate_shape(self) -> None:
        soft = np.eye(4, dtype=np.float32)
        result = aggregate_batches(
            soft,
            [(0, 1, 2, 3)],
            source="test",
        )
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["candidate_count"], 6)
        self.assertIn("W_mean", result["aggregate"])


if __name__ == "__main__":
    unittest.main()

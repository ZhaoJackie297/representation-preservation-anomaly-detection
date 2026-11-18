from __future__ import annotations

import unittest

try:
    import torch

    from coverage_soft_ts2vec import CoverageSoftTS2Vec
except ModuleNotFoundError:
    torch = None
    CoverageSoftTS2Vec = None


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class CoverageSoftTS2VecTest(unittest.TestCase):
    def test_one_batch_step(self) -> None:
        import numpy as np

        model = CoverageSoftTS2Vec(
            input_dims=3,
            output_dims=4,
            hidden_dims=4,
            depth=1,
            device="cpu",
            batch_size=2,
            soft_instance=True,
            soft_temporal=False,
        )
        data = np.random.default_rng(1).normal(size=(4, 8, 3)).astype(np.float32)
        labels = np.eye(4, dtype=np.float32)
        labels[0, 1] = labels[1, 0] = 0.5
        logs = model.fit_coverage_table(
            train_data=data,
            soft_labels=labels,
            batch_table=[(0, 1)],
            n_epochs=1,
        )
        self.assertEqual(len(logs), 1)
        self.assertGreater(logs[0]["active_pair_mass"], 0)


if __name__ == "__main__":
    unittest.main()

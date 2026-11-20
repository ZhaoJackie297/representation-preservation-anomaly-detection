from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from anomaly_transformer_smd import association_terms, my_kl_loss
from omni_anomaly_smd import OmniAnomalyAdapted, gaussian_nll
from protocol_utils import record_run_protocol


class DeepBaselineTest(unittest.TestCase):
    def test_kl_loss_shape(self) -> None:
        left = torch.softmax(torch.randn(4, 10, 10), dim=-1)
        right = torch.softmax(torch.randn(4, 10, 10), dim=-1)
        self.assertEqual(my_kl_loss(left, right).shape, (4,))

    def test_association_terms_shape(self) -> None:
        series = [torch.softmax(torch.randn(2, 4, 10, 10), dim=-1)]
        prior = [torch.softmax(torch.randn(2, 4, 10, 10), dim=-1)]
        series_loss, prior_loss = association_terms(series, prior)
        self.assertEqual(series_loss.shape, (2, 10))
        self.assertEqual(prior_loss.shape, (2, 10))

    def test_omni_adapted_shapes_and_finite_nll(self) -> None:
        model = OmniAnomalyAdapted(input_dim=5, window=16, hidden_dim=8)
        values = torch.randn(3, 16, 5)
        mean, logvar, q_mean, q_logvar = model(values)
        self.assertEqual(mean.shape, values.shape)
        self.assertEqual(logvar.shape, values.shape)
        self.assertEqual(q_mean.shape, (3, 3))
        self.assertEqual(q_logvar.shape, (3, 3))
        self.assertTrue(torch.isfinite(gaussian_nll(values, mean, logvar)).all())

    def test_protocol_merges_resumed_seed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            record_run_protocol(path, {"window": 16}, [1, 2, 3])
            record_run_protocol(path, {"window": 16}, [4, 5])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seeds"], [1, 2, 3, 4, 5])
            self.assertEqual(
                payload["seed_invocations"],
                [{"seeds": [1, 2, 3]}, {"seeds": [4, 5]}],
            )


if __name__ == "__main__":
    unittest.main()

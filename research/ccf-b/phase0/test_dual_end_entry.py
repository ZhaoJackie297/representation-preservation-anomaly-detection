from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dual_end_entry import (
    average_precision,
    choose_batch_table,
    densify,
    evaluate_scores,
    random_batch_table,
    roc_auc,
    robust_scores,
    run,
)


class DualEndEntryTest(unittest.TestCase):
    def test_random_batch_table_is_reproducible_and_disjoint(self) -> None:
        table_a = random_batch_table(
            user_count=20,
            batch_size=4,
            batches=3,
            seed=7,
        )
        table_b = random_batch_table(
            user_count=20,
            batch_size=4,
            batches=3,
            seed=7,
        )
        self.assertEqual(table_a, table_b)
        for batch in table_a:
            self.assertEqual(len(batch), 4)
            self.assertEqual(len(set(batch)), 4)

    def test_batch_table_selector_returns_a_table(self) -> None:
        coverage = SimpleNamespace(
            training_batches=((0, 1, 2, 3), (0, 1, 2, 3))
        )
        official = choose_batch_table(
            variant="A0-official",
            user_count=8,
            batch_size=4,
            seed=3,
            base_steps=2,
            coverage=coverage,
        )
        matched = choose_batch_table(
            variant="A0-matched",
            user_count=8,
            batch_size=4,
            seed=3,
            base_steps=2,
            coverage=coverage,
        )
        r0 = choose_batch_table(
            variant="A1-R0",
            user_count=8,
            batch_size=4,
            seed=3,
            base_steps=2,
            coverage=coverage,
        )
        self.assertEqual(len(official), 2)
        self.assertIsInstance(official[0], tuple)
        self.assertEqual(matched, coverage.training_batches)
        self.assertEqual(len(r0), 2)

    def test_densify_diagonal_is_one(self) -> None:
        values = np.asarray([[0.0, -1.0], [-1.0, 0.0]])
        weights = densify(values, tau=1.0, alpha=0.5)
        np.testing.assert_allclose(np.diag(weights), 1.0)

    def test_metrics_match_known_ranking(self) -> None:
        labels = np.asarray([1, 0, 1, 0])
        scores = np.asarray([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(average_precision(labels, scores), 5.0 / 6.0)
        self.assertAlmostEqual(roc_auc(labels, scores), 0.75)

    def test_event_recall_with_daily_top_k(self) -> None:
        scores = np.asarray(
            [
                [0.9, 0.1, 0.2],
                [0.1, 0.8, 0.2],
                [0.1, 0.2, 0.7],
            ]
        )
        event_id = np.asarray(
            [
                [0, 0, 0],
                [-1, -1, 1],
                [-1, -1, 1],
            ]
        )
        active = np.ones_like(scores, dtype=bool)
        result = evaluate_scores(
            scores=scores,
            event_id=event_id,
            active_mask=active,
            eval_start=0,
            eval_end=2,
            k=1,
        )
        self.assertEqual(result["events_hit"], 2)
        self.assertEqual(result["events_total"], 2)
        self.assertAlmostEqual(result["event_recall_at_k1"], 1.0)

    def test_robust_scores_are_nonnegative(self) -> None:
        train = np.asarray([1.0, 2.0, 3.0, 4.0])
        scores = robust_scores(train, np.asarray([0.0, 3.0, 8.0]))
        self.assertTrue(np.all(scores >= 0.0))

    def test_small_end_to_end_entry_uses_coverage_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            users = np.asarray([f"u{index}" for index in range(16)])
            dates = np.asarray([f"2022-01-{index + 1:02d}" for index in range(8)])
            values = np.random.default_rng(1).normal(
                size=(16, 8, 3)
            ).astype(np.float32)
            active = np.ones((16, 8), dtype=bool)
            event_id = np.full((16, 8), -1, dtype=np.int16)
            np.savez_compressed(
                root / "clue_user_day_dataset.npz",
                users=users,
                dates=dates,
                features=np.asarray(["a", "b", "c"]),
                values=values,
                active_mask=active,
                event_id=event_id,
                train_start_index=np.asarray(0),
                train_end_index=np.asarray(3),
                eval_start_index=np.asarray(4),
                eval_end_index=np.asarray(7),
            )
            np.save(
                root / "clue_profile_features.npy",
                np.random.default_rng(2).normal(size=(16, 2)),
            )
            prior = np.full((16, 16), 1.0 / 15.0)
            np.fill_diagonal(prior, 0.0)
            edge = np.asarray([[0, 1]], dtype=np.int32)
            np.savez_compressed(
                root / "clue_prior_graphs.npz",
                prior=prior,
                A3=edge,
                A5a=edge,
                A5b=edge,
            )
            (root / "clue_profile_prior_metadata.json").write_text(
                json.dumps({"users_order": users.tolist()}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data_dir=root,
                output_dir=root / "output",
                seeds=[7],
                variants=["A1-Cov"],
                batch_size=16,
                epochs=1,
                base_steps=1,
                steps=None,
                tau_temp=1.5,
                device="cpu",
                encode=False,
            )
            summary = run(args)
            model_summary = summary["variants"]["7"]["models"]["A1-Cov"]
            self.assertEqual(model_summary["optimizer_steps"], 1)
            self.assertTrue(model_summary["soft_instance"])
            self.assertTrue(
                (root / "output" / "seed_7" / "A1-Cov" / "model.pth").is_file()
            )


if __name__ == "__main__":
    unittest.main()

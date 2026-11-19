from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from controlled_signal_study import (
    Profile,
    build_alerts,
    causal_context,
    event_metric,
    inject_signal,
    paired_rows,
    robust_transform,
)


class ControlledSignalStudyTest(unittest.TestCase):
    def test_robust_transform_is_nonnegative(self) -> None:
        train = np.asarray([1.0, 2.0, 3.0, 4.0])
        values = np.asarray([0.0, 2.0, 4.0])
        output = robust_transform(train, values)
        self.assertTrue(np.all(output >= 0.0))
        self.assertGreater(output[-1], output[1])

    def test_causal_context_resets_on_inactive_day(self) -> None:
        values = np.asarray([[1.0, 2.0, np.nan, 4.0]])
        active = np.asarray([[True, True, False, True]])
        output = causal_context(values, active, window=3)
        self.assertEqual(output[0, 0], 1.0)
        self.assertEqual(output[0, 1], 1.5)
        self.assertTrue(np.isnan(output[0, 2]))
        self.assertEqual(output[0, 3], 4.0)

    def test_inject_signal_marks_only_target_window(self) -> None:
        values = np.zeros((3, 6, 2), dtype=np.float32)
        active = np.ones((3, 6), dtype=bool)
        channels = np.asarray([0, 1])
        scales = np.asarray([1.0, 1.0])
        modified, labels, event_id, rows = inject_signal(
            values,
            active,
            np.asarray([1]),
            np.asarray([2]),
            signal_type="point",
            strength=1.0,
            channels=channels,
            scales=scales,
        )
        self.assertGreater(modified[1, 2, 0], 0.0)
        self.assertFalse(np.any(modified[0]))
        self.assertTrue(labels[1, 2])
        self.assertEqual(int(event_id[1, 2]), 0)
        self.assertEqual(rows[0]["positive_days"], 1)

    def test_alerts_respect_cooldown(self) -> None:
        score = np.asarray(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        active = np.ones_like(score, dtype=bool)
        alerts = build_alerts(score, active, k=1, cooldown_days=2)
        self.assertEqual(int(alerts[0].sum()), 2)
        self.assertEqual(int(alerts[0, 0]), 1)
        self.assertEqual(int(alerts[0, 2]), 1)

    def test_event_metric_counts_first_hit(self) -> None:
        alerts = np.zeros((2, 8), dtype=np.uint8)
        alerts[0, 3] = 1
        events = pd.DataFrame(
            [
                {"user_index": 0, "start": 2, "end": 4},
                {"user_index": 1, "start": 5, "end": 7},
            ]
        )
        recall, delays = event_metric(
            alerts,
            events,
            np.ones((2, 8), dtype=bool),
            start_index=0,
            end_index=7,
        )
        self.assertEqual(recall, 0.5)
        self.assertEqual(delays[0], 1.0)
        self.assertIsNone(delays[1])

    def test_paired_rows_matches_score_type(self) -> None:
        rows = []
        for method, point_value, context_value in (
            ("control", 1.0, 2.0),
            ("treatment", 3.0, 4.0),
        ):
            for score_type, value in (
                ("point", point_value),
                ("context", context_value),
            ):
                rows.append(
                    {
                        "seed": 1,
                        "signal_type": "point",
                        "strength": 1.0,
                        "split": "test",
                        "method": method,
                        "base_method": method,
                        "score_type": score_type,
                        "pr_auc": value,
                        "roc_auc": value,
                        "event_recall_k5": value,
                        "event_recall_k10": value,
                    }
                )
        profile = Profile(
            data_path=__import__("pathlib").Path("data.npz"),
            model_root=__import__("pathlib").Path("models"),
            model_layout="seed/variant/model",
            train_end_key="train",
            eval_start_key="start",
            eval_end_key="end",
            treatment="treatment",
            control="control",
        )
        output = paired_rows(pd.DataFrame(rows), profile=profile)
        selected = output[
            (output["comparison"] == "treatment_minus_control")
            & (output["metric"] == "pr_auc")
        ]
        self.assertEqual(len(selected), 2)
        self.assertTrue(np.all(selected["mean_delta"] == 2.0))


if __name__ == "__main__":
    unittest.main()

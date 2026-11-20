from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from summarize_cert_baselines import load_summary


class SummarizeCertBaselinesTest(unittest.TestCase):
    def test_load_summary(self) -> None:
        payload = {
            "seed": 20261201,
            "point": {"pr_auc": 0.1, "roc_auc": 0.6},
            "event": {
                "10": {
                    "splits": {
                        "test": {
                            "horizons": {
                                "3": {
                                    "event_recall": 0.25,
                                    "incidents_hit": 6,
                                }
                            }
                        }
                    }
                }
            },
        }
        import json

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = load_summary(path, "baseline")
            self.assertEqual(row["pr_auc"], 0.1)
            self.assertEqual(row["event_hits_k10"], 6)
            self.assertIsNone(row["event_hits_k5"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_official_pai import label_matrix


class OfficialPAITest(unittest.TestCase):
    def test_label_matrix_alignment(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            pd.DataFrame(
                {
                    "user_id": ["u1", "u2"],
                    "date": ["2026-01-02", "2026-01-01"],
                }
            ).to_csv(path, index=False)
            users = np.asarray(["u1", "u2"])
            dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
            matrix = label_matrix(path, users, dates)
            expected = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(matrix, expected)


if __name__ == "__main__":
    unittest.main()

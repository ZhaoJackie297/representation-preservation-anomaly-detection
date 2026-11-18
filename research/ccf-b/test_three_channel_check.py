from __future__ import annotations

import unittest

from three_channel_check import run_algebra_check, run_free_logit_check


class ThreeChannelCheckTest(unittest.TestCase):
    def test_algebra_identities_hold(self) -> None:
        result = run_algebra_check(trials=100, seed=7)
        self.assertLess(result["identity"], 1e-10)
        self.assertLess(result["gradient"], 1e-6)
        self.assertLess(result["diff"], 1e-6)
        self.assertEqual(result["pinsker_violations"], 0)

    def test_free_logit_check_is_small(self) -> None:
        self.assertLess(run_free_logit_check(seed=7), 1e-3)


if __name__ == "__main__":
    unittest.main()

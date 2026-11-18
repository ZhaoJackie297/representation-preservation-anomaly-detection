from __future__ import annotations

import unittest

import numpy as np

from audit_profile_prior_information import (
    profile_prior_information,
    top_m_entropy,
)


class ProfilePriorInformationTest(unittest.TestCase):
    def test_uniform_top_m_is_flat(self) -> None:
        prior = np.full((4, 4), 1.0 / 3.0)
        np.fill_diagonal(prior, 0.0)
        entropy = top_m_entropy(prior, 2)
        np.testing.assert_allclose(entropy, np.log(2.0))

    def test_flat_rule_uses_majority_condition(self) -> None:
        prior = np.full((10, 10), 1.0 / 9.0)
        np.fill_diagonal(prior, 0.0)
        result = profile_prior_information(prior, top_m=5)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["status"], "FLAT_PROFILE_PRIOR")

    def test_peaked_prior_is_informative(self) -> None:
        prior = np.full((5, 5), 1e-12)
        for user in range(5):
            prior[user, (user + 1) % 5] = 1.0
            prior[user] /= prior[user].sum()
        result = profile_prior_information(prior, top_m=1)
        self.assertFalse(result["triggered"])
        self.assertEqual(result["status"], "PRIOR_INFORMATIVE")


if __name__ == "__main__":
    unittest.main()

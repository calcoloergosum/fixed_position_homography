import unittest
import numpy as np
from ransac_test_util import trial_count, confidence


ALMOST_1 = 0.9999999999999999
ALMOST_0 = 0.0000000000000001


class TestRansacTestUtil(unittest.TestCase):
    def test_confidence(self):
        self.assertEqual(confidence(1, 1, 1), 1)
        self.assertEqual(confidence(0, 1, 1), 0)

    def test_trial_count(self):
        try:
            self.assertEqual(trial_count(ALMOST_1, 1, ALMOST_0), 0)
        except AssertionError:
            pass
        try:
            self.assertEqual(trial_count(0.5, 1000000, ALMOST_1), np.inf)
        except AssertionError:
            pass

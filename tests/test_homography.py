import unittest

import numpy as np
from scipy.stats import ortho_group

from homography import calculateHomography, decomposeHomography


class TestHomography(unittest.TestCase):
    def test_homography_estimation(self):
        """
        Test homography estimation by following steps:
        1. randomly compose a ground truth homography
        2. make some points using ground truth
        3. estimate from points
        4. Compare ground truth and estimated homography
        """
        gt = np.random.random((3, 3))
        gt[2, 2] = 1

        pts_1 = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])

        pts_2 = np.concatenate((pts_1.T, np.ones((1, 4))), axis=0).T @ gt.T
        pts_2 /= pts_2[:, np.newaxis, 2]
        pts_2 = pts_2[:, :2]

        M = calculateHomography(pts_1, pts_2)
        np.testing.assert_almost_equal(M, gt)

    def test_homography_decomposition(self):
        """
        Test homography decomposition by following steps:
        1. prepare affine with random rotation R and translation t
        2. make homography of affine from step 1, with random projection P
          **NOTE: t first, then R !! ***
        3. decompose homography using P to get R' and t'
        4. compare (R, t) and (R', t')
        """
        # step 1
        R = ortho_group.rvs(dim=3)
        R /= np.linalg.det(R)
        t = np.random.random((3,)) * 2 - 1
        P = np.random.random((3, 3))
        P[0, 0] = P[1, 1] = P[2, 2] = 1
        P[0, 1:] = 0
        P[1, 2] = 0

        # step 2
        A = np.array([R[:, 0], R[:, 1], R @ t]).T
        H = P @ A

        # step 3
        R_, t_ = decomposeHomography(H, P)

        # step 4
        np.testing.assert_almost_equal(R, R_)
        np.testing.assert_almost_equal(t, t_)

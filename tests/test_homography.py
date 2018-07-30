import unittest

import numpy as np
from scipy.stats import ortho_group

from homography import Homography, ProjectionModel
from ransac_test_util import trial_count


class TestHomography(unittest.TestCase):
    def test_homography_estimation_exact(self):
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

        data = np.concatenate((pts_1, pts_2), axis=1)
        M = ProjectionModel().fit(data).M
        np.testing.assert_almost_equal(M, gt)

    def test_homography_estimation_overconstraint(self):
        """
        Test homography estimation by following steps:
        1. randomly compose a ground truth homography
        2. make some points using ground truth
        3. estimate from points
        4. Compare ground truth and estimated homography
        """
        N = 1000
        assert N > 4

        gt = np.random.random((3, 3))
        gt[2, 2] = 1

        pts_1 = np.random.rand(N, 2)

        pts_2 = np.concatenate((pts_1.T, np.ones((1, N))), axis=0).T @ gt.T
        pts_2 /= pts_2[:, np.newaxis, 2]
        pts_2 = pts_2[:, :2]

        data = np.concatenate((pts_1, pts_2), axis=1)
        M = ProjectionModel().fit(data).M
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
        hom = Homography(H, P)
        R_, t_ = hom.R, hom.t

        # step 4
        np.testing.assert_almost_equal(R, R_)
        np.testing.assert_almost_equal(t, t_)

    def test_ransac(self):
        """
        Test ransac by following steps:
        1. make random homography
        2. make inliers with noise
        3. make outliers
        4. RUN
        """
        N_TOTAL = 1000
        N_INLIER = 500
        N_ITER = trial_count(N_INLIER / N_TOTAL, 4)
        N_OUTLIER = N_TOTAL - N_INLIER
        NOISE_SIZE = 0

        # step 1
        gt = np.random.random((3, 3))
        gt[2, :] *= 100
        gt[2, 2] = 1

        # step 2
        hom = Homography(gt)
        x = np.random.random((N_INLIER, 2))
        y = hom.transform(x)

        x += np.random.random((N_INLIER, 2)) * NOISE_SIZE
        y += np.random.random((N_INLIER, 2)) * NOISE_SIZE
        inliers = np.concatenate((x, y), axis=1)

        # step 3
        outliers = np.random.random((N_OUTLIER, 4))
        data = np.concatenate(
            (inliers, outliers),
            axis=0
        )

        # step 4
        from ransac import RANSAC
        ransac_inst = RANSAC(
            n_sample=4,
            n_iter=N_ITER,
            err_thres=0.1,
        )
        model = ProjectionModel()
        best_func, best_inliers = ransac_inst(data, model)
        np.testing.assert_almost_equal(best_func.M, gt)

        # # step 5
        # from ransac import MSAC
        # msac_inst = MSAC(
        #     n_sample=4,
        #     n_iter=N_ITER,
        #     err_thres=0.01,
        # )
        # model = ProjectionModel()
        # best_func, best_inliers = msac_inst(data, model)
        # np.testing.assert_almost_equal(best_func.M, gt)

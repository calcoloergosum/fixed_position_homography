import unittest

import numpy as np
import scipy.linalg

from regression_model import RegressionModel, Function
import ransac


class LinearFunctionL2(Function):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self._o_dim, self._i_dim = matrix.shape

    def error(self, data):
        assert len(data.shape) == 2
        assert data.shape[1] == self._o_dim + self._i_dim
        x = data[:, :self._i_dim]
        y = data[:, self._i_dim:]
        return np.sum((y.T - self.matrix @ x.T) ** 2, axis=0)


class LinearLeastSquaresRegression(RegressionModel):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, data: np.ndarray) -> LinearFunctionL2:
        assert len(data.shape) == 2
        assert data.shape[1] == self.input_dim + self.output_dim
        A = data[:, :self.input_dim]
        B = data[:, self.input_dim:]
        x, resids, rank, s = scipy.linalg.lstsq(A, B)
        assert x.shape == (self.input_dim, self.output_dim)
        return LinearFunctionL2(x.T)


class test_RANSAC(unittest.TestCase):
    def setUp(self):
        """
        According to "CHOI, KIM, YU: PERFORMANCE EVALUATION OF RANSAC FAMILY",
        robustness difference between RANSAC/MSAC and MLESAC is apparent when mixture ratio < 0.4
        """
        dim_in = 3
        dim_out = 3
        self.dim_in, self.dim_out = dim_in, dim_out

        inlier_noise = 0.001
        inlier_ratio = 0.5
        total_size = 200
        inlier_size = int(total_size * inlier_ratio)
        outlier_size = total_size - inlier_size
        self.inlier_ratio = inlier_ratio

        ground_truth = np.random.rand(dim_out, dim_in)
        self.ground_truth = ground_truth

        inlier_x = np.random.rand(inlier_size, dim_in)
        inlier_y = ground_truth @ inlier_x.T + np.random.normal(0, inlier_noise, (dim_out, inlier_size))
        inliers = np.concatenate((inlier_x, inlier_y.T), axis=1)
        assert inliers.shape == (inlier_size, dim_in + dim_out)

        outlier_x = np.random.rand(outlier_size, dim_in)
        outlier_y = np.random.rand(outlier_size, dim_out)
        outliers = np.concatenate((outlier_x, outlier_y), axis=1)
        assert outliers.shape == (outlier_size, dim_in + dim_out)

        data = np.concatenate((inliers, outliers), axis=0)
        assert data.shape == (inlier_size + outlier_size, dim_in + dim_out)
        np.random.shuffle(data)

        self.data = data
        self.model = LinearLeastSquaresRegression(self.dim_in, self.dim_out)

    def test_mlesac(self):
        mlesac = ransac.MLESAC(
            n_sample=2,
            n_iter=1000,
            sigma=0.01,
            n_EM_iter=3,
            dim=3,
        )

        best_func, inlier_prob = mlesac(self.data, self.model)
        inliers = self.data[inlier_prob > 0]
        np.testing.assert_almost_equal(len(inliers) / len(self.data), self.inlier_ratio, decimal=3)
        np.testing.assert_array_almost_equal(best_func.matrix, self.ground_truth, decimal=3)
        print(
            '[*] mlesac error: {error}'.format(
                error=np.sqrt(((best_func.matrix - self.ground_truth) ** 2).sum())
            )
        )

    def test_ransac(self):
        ransac_instance = ransac.RANSAC(
            n_sample=5,
            n_iter=10000,
            err_thres=0.01,
            inlier_ratio=0.2
        )

        best_func, best_inliers = ransac_instance(self.data, self.model)
        np.testing.assert_almost_equal(len(best_inliers) / len(self.data), self.inlier_ratio, decimal=3)
        np.testing.assert_array_almost_equal(best_func.matrix, self.ground_truth, decimal=3)
        print(
            '[*] ransac error: {error}'.format(
                error=np.sqrt(((best_func.matrix - self.ground_truth) ** 2).sum())
            )
        )

    def test_msac(self):
        msac_instance = ransac.MSAC(
            n_sample=5,
            n_iter=10000,
            err_thres=0.01,
            inlier_ratio=0.2
        )

        best_func, best_inliers = msac_instance(self.data, self.model)
        np.testing.assert_almost_equal(len(best_inliers) / len(self.data), self.inlier_ratio, decimal=3)
        np.testing.assert_array_almost_equal(best_func.matrix, self.ground_truth, decimal=3)
        print(
            '[*] msac error: {error}'.format(
                error=np.sqrt(((best_func.matrix - self.ground_truth) ** 2).sum())
            )
        )

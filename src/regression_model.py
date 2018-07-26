import numpy as np


class Function:
    def error(self, inliers, outliers=[]):
        raise NotImplementedError()


class RegressionModel:
    def fit(self, inliers: np.ndarray) -> Function:
        """Returns Function"""
        raise NotImplementedError()

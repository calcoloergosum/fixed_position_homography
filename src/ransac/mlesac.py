from functools import reduce
from operator import mul

import numpy as np


class MLESAC:
    class Partition:
        def __init__(self, inliers, outliers):
            self.inliers = inliers
            self.outliers = outliers

    def __init__(self, n_sample, n_iter, sigma, n_EM_iter, dim):
        self.n_sample = 10
        self.n_iter = n_iter
        self.sigma = sigma
        self.n_EM_iter = n_EM_iter
        self.dim = dim

        self._norm_reg = (np.sqrt(2 * np.pi) * self.sigma)

    def __call__(
        self,
        data,
        model,
    ):
        n_data = len(data)
        assert data.shape[1] == 2 * self.dim

        best_score = np.inf
        best_assign = np.zeros(n_data)
        best_function = None

        diameter = self._diameter(data)
        for mask_idxs in self.random_mask_indices(n_data):

            mixture_ratio = 0.5  # inlier / total
            inliers = data[mask_idxs]

            for _ in range(self.n_EM_iter):
                f = model.fit(inliers)
                errors = f.error(data)
                p_i = mixture_ratio * self.norm(errors, self.sigma)
                p_o = (1 - mixture_ratio) / diameter

                inlier_prob = p_i / (p_i + p_o)
                mixture_ratio = np.mean(inlier_prob)
                mask_idxs[inlier_prob > 0.95] = True
                inliers = data[mask_idxs]
            errors = f.error(data)
            likelihood = -1 * np.sum(np.log(
                mixture_ratio * self.norm(errors, self.sigma) + p_o
            ))
            if likelihood < best_score:
                best_score = likelihood
                best_assign = inliers
                best_function = f
        return best_function, best_assign

    def norm(self, errors, sigma):
        return np.exp(-1 * errors ** 2 / 2 / self.sigma ** 2) / (self._norm_reg ** (2 * self.dim))

    def _diameter(self, data):
        return reduce(mul, (np.max(data, axis=0) - np.min(data, axis=0)))

    def random_mask_indices(self, n_data):
        for i in range(self.n_iter):
            # Random Sample
            mask = np.zeros((n_data,), dtype=bool)
            _idxs = np.random.choice(n_data, self.n_sample)

            mask[_idxs] = True
            mask[~_idxs] = False

            yield mask

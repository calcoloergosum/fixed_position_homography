from functools import reduce
from operator import mul

import numpy as np

from .ransac import RANSACFamily, Partition


class MLESAC(RANSACFamily):
    def __init__(self, n_sample, n_iter, sigma, n_EM_iter, dim):

        super().__init__(n_sample, n_iter)

        self.dim = dim
        self.sigma = sigma
        self.n_EM_iter = n_EM_iter

        self._norm_reg = (np.sqrt(2 * np.pi) * self.sigma)
        self._diameter = None

    def __call__(self, data, model):
        self._diameter = self._get_diameter(data)
        x = super().__call__(data, model)
        return x

    def update_partition(self, model, data, mask_idxs):
        """
        Update Partition
        Note that mask_idxs gets dirty after processing
        """
        inliers = data[mask_idxs]

        mixture_ratio = 0.5  # inlier / total
        for _ in range(self.n_EM_iter):
            f = model.fit(inliers)
            errors = f.error(data)
            p_i = mixture_ratio * self.norm(errors, self.sigma)
            p_o = (1 - mixture_ratio) / self._diameter

            inlier_prob = p_i / (p_i + p_o)
            mixture_ratio = np.mean(inlier_prob)
            updated_idxs = np.logical_xor(inlier_prob > 0.95, mask_idxs)
            inliers = data[np.logical_or(mask_idxs, updated_idxs)]
            if not np.any(updated_idxs):
                break
        else:
            f = model.fit(inliers)
            errors = f.error(data)

        likelihood = -1 * np.sum(np.log(
            mixture_ratio * self.norm(errors, self.sigma) + p_o
        ))
        return Partition(
            model.fit(inliers),
            inliers,
            data[~mask_idxs],
            error=likelihood
        )

    def norm(self, errors, sigma):
        return np.exp(-1 * errors ** 2 / 2 / self.sigma ** 2) / (self._norm_reg ** (2 * self.dim))

    def _get_diameter(self, data):
        return reduce(mul, (np.max(data, axis=0) - np.min(data, axis=0)))

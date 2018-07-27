# Naive RANSAC (Random Sample Consensus)

import numpy as np

from regression_model import RegressionModel, Function


class Partition:
    def __init__(self, function: Function, inliers: np.ndarray, outliers: np.ndarray, error):
        self.inliers = inliers
        self.outliers = outliers
        self.function = function
        self.error = error

    def is_better_than(self, other_record):
        return self.error < other_record.error


class RANSACFamily:
    def __init__(self, n_sample, n_iter):
        self._n_sample = n_sample
        self._n_iter = n_iter

    def __call__(self, data, model: RegressionModel):
        n_data = len(data)

        best_partition = Partition(function=None, inliers=[], outliers=data, error=np.inf)
        for mask_idxs in self.random_mask_indices(n_data):
            partition = self.update_partition(model, data, mask_idxs)

            if partition.is_better_than(best_partition):
                best_partition = partition

        return (
            best_partition.function,
            best_partition.inliers
        )

    def random_mask_indices(self, n_data):
        for i in range(self._n_iter):
            # Random Sample
            mask = np.zeros((n_data,), dtype=bool)
            _idxs = np.random.choice(n_data, self._n_sample, replace=False)

            mask[_idxs] = True
            yield mask


class RANSAC(RANSACFamily):
    """More Inlier the better"""
    def __init__(self, n_sample, n_iter, err_thres):
        self.err_thres = err_thres
        super().__init__(n_sample, n_iter)

    def update_partition(self, model, data, mask_idxs):
        """Identity"""
        inliers = data[mask_idxs]
        others = data[~mask_idxs]

        function = model.fit(inliers)
        error = function.error(others)
        threshold_mask = error < self.err_thres
        new_inliers = others[threshold_mask]
        inliers = np.concatenate(
            (
                inliers,
                new_inliers
            )
        )
        outliers = others[~threshold_mask]
        return Partition(
            function,
            inliers,
            outliers,
            error=np.inf if function is None else self.err_thres * len(outliers),
        )

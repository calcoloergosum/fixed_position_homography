# Naive RANSAC (Random Sample Consensus)

import numpy as np
from tqdm import tqdm

from regression_model import RegressionModel, Function


class _SamplePartition:
    def __init__(self, inliers: np.ndarray, others: np.ndarray):
        self.inliers = inliers
        self.others = others

    @staticmethod
    def random(data, n_sample, n_data=None, mask=None):
        # mask is given because of reallocation problem
        if not n_data:
            n_data = len(data)
        _idxs = np.random.choice(n_data, n_sample)

        mask[_idxs] = True
        sample = _SamplePartition(
            inliers=data[mask],
            others=data[~mask],
        )
        mask[_idxs] = False
        return sample


class _Partition:
    @staticmethod
    def _get_error(function, inliers, outliers):
        raise NotImplementedError()

    def __init__(self, function: Function, inliers: np.ndarray, outliers: np.ndarray):
        self.inliers = inliers
        self.outliers = outliers
        self.function = function
        self.error = self._get_error(
            self.function, self.inliers, self.outliers)

    def is_better_than(self, other_record):
        return self.error < other_record.error


class _RANSACFamily:
    def __init__(self, n_sample, n_iter, err_thres, inlier_ratio, partition_class: _Partition):
        self._iter = n_iter
        self._threshold = err_thres
        self._inlier_ratio = inlier_ratio
        self._n_sample = n_sample
        self._partition_class = partition_class

    def __call__(self, data, model: RegressionModel):
        n_data = len(data)
        mask = np.zeros((n_data,), dtype=bool)

        best_partition = self._partition_class(
            function=None, inliers=[], outliers=data)
        for i in tqdm(range(self._iter)):
            sampled_partition = _SamplePartition.random(
                data,
                n_sample=self._n_sample,
                n_data=n_data,
                mask=mask,
            )

            sampled_function = model.fit(sampled_partition.inliers)
            others_err = sampled_function.error(sampled_partition.others)
            newly_found_inliers = sampled_partition.others[others_err <
                                                           self._threshold]

            inlier_size = len(sampled_partition.inliers) + \
                len(newly_found_inliers)
            if inlier_size > self._inlier_ratio * n_data:
                inliers = np.concatenate(
                    (sampled_partition.inliers, newly_found_inliers))
                outliers = sampled_partition.others[others_err >=
                                                    self._threshold]
                this_partition = self._partition_class(
                    model.fit(inliers),
                    inliers,
                    outliers,
                )
                if this_partition.is_better_than(best_partition):
                    best_partition = this_partition

        if best_partition.function is None:
            raise ValueError("Couldn't find a function with inlier_ratio higher than {}".format(
                self._inlier_ratio
            ))

        return (
            best_partition.function,
            best_partition.inliers
        )


class RANSAC(_RANSACFamily):
    """More Inlier the better"""

    def __init__(self, n_sample, n_iter, err_thres, inlier_ratio):

        class _RANSAC_Partition(_Partition):
            @staticmethod
            def _get_error(function, inliers, outliers):
                if function is None:
                    return np.inf
                else:
                    return err_thres * len(outliers)

        super().__init__(n_sample, n_iter, err_thres, inlier_ratio, _RANSAC_Partition)


class MSAC(_RANSACFamily):
    """More Inlier Fit the better"""

    def __init__(self, n_sample, n_iter, err_thres, inlier_ratio):

        class _MSAC_Partition(_Partition):
            @staticmethod
            def _get_error(function, inliers, outliers):
                if function is None:
                    return np.inf
                else:
                    return np.mean(function.error(inliers)) + err_thres * len(outliers)

        super().__init__(n_sample, n_iter, err_thres, inlier_ratio, _MSAC_Partition)

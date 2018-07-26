# Naive RANSAC (Random Sample Consensus)

import numpy as np

from regression_model import RegressionModel, Function


class _SamplePartition:
    def __init__(self, inliers: np.ndarray, others: np.ndarray):
        self.inliers = inliers
        self.others = others

    @staticmethod
    def random(data, n_sample, n_data=None):
        if not n_data:
            n_data = len(data)
        mask = np.zeros((n_data,), dtype=bool)
        _idxs = np.random.choice(n_data, n_sample, replace=False)

        mask[_idxs] = True
        sample = _SamplePartition(
            inliers=data[mask],
            others=data[~mask],
        )
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

    def update_partition(self, function, partition):
        raise NotImplementedError

    def __call__(self, data, model: RegressionModel):
        n_data = len(data)

        best_partition = self._partition_class(
            function=None, inliers=[], outliers=data)
        for _ in range(self._iter):
            sampled_partition = _SamplePartition.random(
                data,
                n_sample=self._n_sample,
                n_data=n_data,
            )

            updated_partition = self.update_partition(model, sampled_partition)
            if updated_partition.is_better_than(best_partition):
                best_partition = updated_partition

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

    def __init__(self, n_sample, n_iter, err_thres, inlier_ratio=0.5):

        class _RANSAC_Partition(_Partition):
            @staticmethod
            def _get_error(function, inliers, outliers):
                if function is None:
                    return np.inf
                else:
                    return err_thres * len(outliers)

        super().__init__(n_sample, n_iter, err_thres, inlier_ratio, _RANSAC_Partition)

    def update_partition(self, model, partition):
        """Identity"""
        function = model.fit(partition.inliers)
        error = function.error(partition.others)
        threshold_mask = error < self._threshold
        new_inliers = partition.others[threshold_mask]
        inliers = np.concatenate(
            (
                partition.inliers,
                new_inliers
            )
        )
        outliers = partition.others[~threshold_mask]
        return self._partition_class(
            function,
            inliers,
            outliers,
        )


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

    def update_partition(self, model, partition):
        """Update Partition"""
        function = model.fit(partition.inliers)
        n_inliers = len(partition.inliers)
        n_data = n_inliers + len(partition.others)
        others_err = function.error(partition.others)
        newly_found_inliers = partition.others[others_err <
                                               self._threshold]
        inlier_size = n_inliers + \
            len(newly_found_inliers)
        if inlier_size > self._inlier_ratio * n_data:
            inliers = np.concatenate(
                (partition.inliers, newly_found_inliers))
            outliers = partition.others[others_err >=
                                        self._threshold]
            return self._partition_class(
                model.fit(inliers),
                inliers,
                outliers,
            )
        return self._partition_class(
            function,
            partition.inliers,
            partition.others
        )

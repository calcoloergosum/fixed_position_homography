import numpy as np

from .ransac import RANSACFamily, Partition


class MSAC(RANSACFamily):
    """More Inlier Fit the better"""

    def __init__(self, n_sample, n_iter, err_thres, local_iter):
        self.local_iter = local_iter
        self.err_thres = err_thres
        super().__init__(n_sample, n_iter)

    def update_partition(self, model, data, mask_idxs):
        """Update Partition"""
        inliers = data[mask_idxs]
        others = data[~mask_idxs]
        for _ in range(self.local_iter):
            function = model.fit(inliers)

            others_err = function.error(others)
            new_inliers = others[others_err < self.err_thres]
            if len(new_inliers) == 0:
                break
            inliers = np.concatenate((inliers, new_inliers), axis=0)
            others = others[others_err >= self.err_thres]

        return Partition(
            model.fit(inliers),
            inliers,
            others,
            error=np.inf if function is None else np.mean(function.error(inliers)) + self.err_thres * len(others),
        )

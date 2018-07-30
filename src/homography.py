import numpy as np
from regression_model import RegressionModel, Function


class ProjectionModel(RegressionModel):
    def __init__(self, intrinsic=None, t_weight=1):
        """
        t_weight: translation weight
        """
        if intrinsic is None:
            intrinsic = np.identity(3)
        self.intrinsic = intrinsic
        self._intrinsic_inv = np.linalg.inv(self.intrinsic)
        self.t_weight = t_weight

    def fit(self, data):
        """
        Given two point sets from view 1 and view 2, find homography from view 1 to view 2
        When data length is 4, use exact solution
        Otherwise use least squares minimization
        """
        assert len(data) >= 4

        # apply intrinsic
        pts1, pts2 = data[:, :2], data[:, 2:]

        pts1 = np.concatenate((pts1, np.ones((len(data), 1))), axis=1)
        pts2 = np.concatenate((pts2, np.ones((len(data), 1))), axis=1)
        pts2, pts2 = (self._intrinsic_inv @ pts1.T).T, (self._intrinsic_inv @ pts2.T).T
        pts1 = pts1[:, :2] / pts1[:, 2, np.newaxis]
        pts2 = pts2[:, :2] / pts2[:, 2, np.newaxis]

        ps = []
        for pt1, pt2 in zip(pts1, pts2):
            x1, y1 = pt1
            x2, y2 = pt2

            p = [
                (-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2),
                (0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2),
            ]
            ps.extend(p)
        A = np.array(ps)
        ps = np.array(ps)

        # exact
        u, s, v = np.linalg.svd(A)
        solution = v[-1]
        solution /= solution[-1]
        return Homography(solution.reshape(3, 3), self.intrinsic)


class Homography(Function):
    def __init__(self, M, intrinsic=None):
        if intrinsic is None:
            intrinsic = np.identity(3)
        self.intrinsic = intrinsic
        self._intrinsic_inv = np.linalg.inv(self.intrinsic)

        self.M = M
        self._inv_M = np.linalg.inv(M)

        self.R, self.t = self._decompose()

    def error(self, inliers, outliers=None):
        """
        This is actually somewhat different from first-order estimation by Hartley and Sturm.
        if you are reading this, please implement :D.
        """
        x = inliers[:, :2]
        y = inliers[:, 2:]

        # apply intrinsic
        x = np.concatenate((x, np.ones((len(inliers), 1))), axis=1)
        y = np.concatenate((y, np.ones((len(inliers), 1))), axis=1)
        x, y = (self._intrinsic_inv @ x.T).T, (self._intrinsic_inv @ y.T).T
        x = x[:, :2] / x[:, 2, np.newaxis]
        y = y[:, :2] / y[:, 2, np.newaxis]

        x_ = self.transform(y, inverse=True)
        y_ = self.transform(x)

        return np.sqrt(np.sum(
            (0.5 * (y_ - y)) ** 2 +
            (0.5 * (x - x_)) ** 2,
            axis=1
        ))

    def transform(self, pts, inverse=False):
        pts_affine = np.concatenate((pts, np.ones((len(pts), 1))), axis=1)
        if not inverse:
            warped = (self.M @ pts_affine.T).T
        else:
            warped = (self._inv_M @ pts_affine.T).T
        warped /= warped[:, 2].reshape(-1, 1)
        return warped[:, :2]

    def _decompose(self):
        """Decompose homography to rotation and transition, provided TRANSITION IS APPLIED FIRST"""
        A = self._intrinsic_inv @ self.M

        r1 = A[:, 0]
        r2 = A[:, 1]
        t_rot = A[:, 2]

        r3 = np.cross(r1, r2)
        r3 /= np.sqrt(np.linalg.norm(r3))
        R_ = np.array([r1, r2, r3]).T
        t_ = R_.T @ t_rot

        return R_, t_

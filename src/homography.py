import numpy as np
from regression_model import RegressionModel, Function


class ProjectionModel(RegressionModel):
    def __init__(self, intrinsic=None):
        if intrinsic is None:
            intrinsic = np.identity(3)
        self.intrinsic = intrinsic

    def fit(self, data):
        """
        Given two point sets from view 1 and view 2, find homography from view 1 to view 2
        When data length is 4, use exact solution
        Otherwise use least squares minimization
        """
        pts_1, pts_2 = data[:, :2], data[:, 2:]
        ps = []
        for pt1, pt2 in zip(pts_1, pts_2):
            x1, y1 = pt1
            x2, y2 = pt2

            p = [
                (-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2),
                (0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2),
            ]
            ps.extend(p)
        ps.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        A = np.array(ps)
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        ps = np.array(ps)

        # exact
        if len(data) == 4:
            solution = np.linalg.solve(A, b)
        # least squares
        else:
            solution = np.linalg.pinv(A.T @ A) @ A.T @ b
        return Homography(solution.reshape(3, 3), self.intrinsic)


class Homography(Function):
    def __init__(self, M, intrinsic=None):
        if intrinsic is None:
            intrinsic = np.identity(3)
        self.intrinsic = intrinsic
        self.M = M
        self.R, self.t = Homography.decomposeHomography(M, intrinsic)
        self._inv_M = np.linalg.inv(M)

    def error(self, inliers, outliers=None):
        x = inliers[:, :2]
        y = inliers[:, 2:]

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

    @staticmethod
    def decomposeHomography(M, intrinsic):
        """Decompose homography to rotation and transition, provided TRANSITION IS APPLIED FIRST"""
        if intrinsic is not None:
            A = np.linalg.inv(intrinsic) @ M
        else:
            A = M

        r1 = A[:, 0]
        r2 = A[:, 1]
        t_rot = A[:, 2]

        r3 = np.cross(r1, r2)
        r3 /= np.sqrt(np.linalg.norm(r3))
        R_ = np.array([r1, r2, r3]).T
        t_ = R_.T @ t_rot

        return R_, t_

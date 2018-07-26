import numpy as np


def calculateHomography(pts_1, pts_2):
    """Given two point sets from view 1 and view 2, find homography from view 1 to view 2"""
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
    b = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ps = np.array(ps)
    solution = np.linalg.solve(A, b)
    return solution.reshape(3, 3)


def decomposeHomography(M, intrinsic=None):
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

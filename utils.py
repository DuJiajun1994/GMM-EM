import numpy as np
import math


def gauss(x, u, cov):
    diff = np.mat(x - u)
    cov = np.mat(cov)
    prob = np.exp(- np.array((diff * cov.I * diff.T))[0, 0] / 2) / (2 * math.pi * math.sqrt(np.linalg.det(cov)))
    return prob


def compute_prob(x, c, u, cov):
    class_num = len(c)
    prob = np.array([
        gauss(x, u[cls_id], cov[cls_id]) for cls_id in range(class_num)
    ])
    prob = prob * c
    return prob

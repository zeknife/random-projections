import numpy as np
from math import sqrt

class RP:
    def transform(self, x):
        """Transform a data array of shape (num samples, num features), returning array of shape (num samples, num reduced features)"""
        return (self.mapping @ x.T).T

    def fast_inverse(self, x):
        """Inverse of self.transform, using transpose as approximation of true inverse"""
        return (self.mapping.T @ x.T).T

    def inverse(self, x):
        """Inverse of self.transform, using pseudo-inverse as a better approximation of true inverse"""
        return (np.linalg.pinv(self.mapping) @ x.T).T


class GaussianRP(RP):
    def __init__(self, n_after, n_before):
        rng = np.random.default_rng()
        rp = rng.standard_normal((n_after, n_before))
        norms = np.linalg.norm(rp, axis=0)
        self.mapping = rp/norms # What if we don't normalize the columns? Well it would affect the distance measure in some random fashion...
        self.dist_factor = sqrt(n_before/n_after)

class OrthoRP(RP):
    def __init__(self, n_after, n_before):
        rng = np.random.default_rng()
        rp = rng.standard_normal((n_before, n_before))
        q,r = np.linalg.qr(rp)
        rp = q[:n_after,:]
        norms = np.linalg.norm(rp, axis=0)
        self.mapping = rp/norms
        self.dist_factor = sqrt(n_before/n_after)

class SparseRP(RP):
    def __init__(self, n_after, n_before):
        rp = np.random.choice(a=[-1,0,1], size=(n_after, n_before), p=[1/6,2/3,1/6])
        norms = np.linalg.norm(rp, axis=0)
        self.mapping = np.nan_to_num(rp/norms) # Set nan values to 0
        self.dist_factor = sqrt(n_before/n_after)
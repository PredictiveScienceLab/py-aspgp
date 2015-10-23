"""
Define a prior for the Stiefel manifold.

Author:
    Ilias Bilionis

Date:
    4/4/2015

"""


import GPy
import numpy as np
from numpy.random import randn
from scipy.linalg import orth
import scipy.stats
from scipy.stats import norm


__all__ = ['StiefelPrior', 'UninformativePrior']


class StiefelPrior(GPy.priors.Prior):
    """
    A prior on the space of Stiefel manifolds.
    """
    domain = GPy.priors._REAL
    _instance = None

    def __init__(self, input_dim, active_dim, alpha=10.):
        """
        Initialize the object.
        """
        self.alpha = alpha
        self._in = input_dim
        self._ad = active_dim
        self.input_dim = input_dim * active_dim

    def __str__(self):
        return 'Stiefel({0:d},{1:d})'.format(self._in, self._ad)

    def lnpdf(self, W):
        return 0.5 * self.alpha * W ** 2

    def lnpdf_grad(self, W):
        return -self.alpha * W

    def rvs(self, n):
        W = np.ndarray((n, self.input_dim))
        for i in xrange(n):
            W[i, :] = orth(randn(self._in, self._ad)).flatten()
        return W


class UninformativePrior(GPy.priors.Prior):
    """
    A simple uninformative prior.
    """

    def __init__(self, input_dim=1):
        """
        Initialize the object.
        """
        self.input_dim = input_dim

    def __str__(self):
        return 'Uninformative({0:d})'.format(self.input_dim)

    def lnpdf(self, x):
        return -np.log(x)

    def lnpdf_grad(self, x):
        return -1. / x

    def rvs(self, n):
        return 1.

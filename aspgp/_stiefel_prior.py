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

    def __init__(self, num_rows, num_cols, alpha=0.,
                 fixed_cols=0):
        """
        Initialize the object.

        :param num_rows:        The number of rows.
        :param num_cols:        The number of columns.
        :param alpha:           The precision of a Gaussian conditioned to
                                stay on the Stiefel manifold.
        :param fixed_cols:      The number of identity columns to be attached
                                at the end.
        """
        self.alpha = alpha
        self.D = num_rows
        self.d = num_cols
        self.k = fixed_cols

    def __str__(self):
        return 'Stiefel({0:d},{1:d})'.format(self.D, self.d)

    def lnpdf(self, W):
        return -0.5 * self.alpha * W ** 2

    def lnpdf_grad(self, W):
        D = self.D
        d = self.d
        k = self.k
        G = -self.alpha * W.reshape((D, d))
        if k >= 1:
            G[:-k, -k:] = 0.
            G[-k:, :-k] = 0.
            G[-k:, -k:] = np.zeros((k, k))
        return G.flatten()

    def rvs(self, n=1):
        """
        It ignores the dimension ``n``.
        """
        D = self.D
        d = self.d
        k = self.k
        W_sub = orth(randn(D - k, d - k))
        return np.vstack([np.hstack([W_sub, np.zeros((D - k, k))]),
                        np.hstack([np.zeros((k, d - k)), np.eye(k)])]).flatten()


class UninformativePrior(GPy.priors.Prior):
    """
    A simple uninformative prior.
    """

    def __init__(self, num_rows=1):
        """
        Initialize the object.
        """
        self.num_rows = num_rows

    def __str__(self):
        return 'Uninformative({0:d})'.format(self.num_rows)

    def lnpdf(self, x):
        return -np.log(x)

    def lnpdf_grad(self, x):
        return -1. / x

    def rvs(self):
        return 1.

"""
Implements a classic active subspace GP model.

Author:
    Ilias Bilionis

Date:
    4/15/2015

"""


from . import ActiveSubspaceGPRegression
from scipy.linalg import svd


__all__ = ['ClassicActiveSubspaceGPRegression']


class ClassicActiveSubspaceGPRegression(ActiveSubspaceGPRegression):
    
    """
    The object builds on the functionality of ``ActiveSubspaceGPRegression.
    """

    # The observed gradients
    _G = None

    @property
    def G(self):
        """
        :getter: the observed gradients.
        """
        return self._G

    def __init__(self, X, Y, G, inner_kernel, W=None, **kwargs):
        """
        Initialize the model.
        """
        super(ClassicActiveSubspaceGPRegression, self).__init__(X, Y, inner_kernel, W=W,
                                                                **kwargs)
        assert X.shape[0] == G.shape[0]
        assert X.shape[1] == G.shape[1]
        self._G = G
        # Compute W using SVD
        U, s, V = svd(self.G)
        self.kern.W = V[:self.kern.active_dim, :].T
        self.kern.W.fix()

    def optimize(self, **kwargs):
        """
        The options are the same as those of the classic ``GPRegression.optimize()``.
        """
        super(ActiveSubspaceGPRegression, self).optimize(**kwargs)

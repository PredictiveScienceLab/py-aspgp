"""
A kernel with built in dimensionality reduction
for active subspaces based GP's.

Author:
    Ilias Bilionis
    Predictive Science Laboratory,
    School of Mechanical Engineering,
    Purdue University,
    West Lafayette, IN, USA

Date:
    12/27/2014

"""

#add packages
from GPy.kern import Kern
from GPy.kern import RBF
from GPy.core.parameterization import Param
from GPy.util.caching import Cache_this
import numpy as np
from numpy.random import randn
from scipy.linalg import orth
from . import StiefelPrior


__all__ = ['ActiveSubspaceKernel']

#define the active subspace kernel class
class ActiveSubspaceKernel(Kern):

    """
    A kernel of the following form:

        .. math::

            k(x, x') = k_0(Wx, Wx')

    """

    _inner_kernel = None

    @property
    def inner_kernel(self):
        """
        :getter: Get the inner kernel.
        """
        return self._inner_kernel

    @inner_kernel.setter
    def inner_kernel(self, value):
        """
        :setter: Set the inner kernel.
        """
        assert isinstance(value, Kern), 'The inner kernel must be a'\
               + ' proper `Gpy.kern.Kern` object.'
        assert value.input_dim <= self.input_dim, 'The number of active'\
               + ' dimensions must be smaller than or equal to the number'\
               + ' of inputs.'
        self._inner_kernel = value

    @property
    def active_dim(self):
        """
        :getter: Get the number of active dimensions.
        """
        return self.inner_kernel.input_dim

    def __init__(self, input_dim, inner_kernel, W=None,
                 name='ActiveSubspaceKernel'):
        """
        Initialize the object.
        """
        super(ActiveSubspaceKernel, self).__init__(input_dim, None, name,
                                                   useGPU=False)
        self.inner_kernel = inner_kernel
        if W is None:
            W = randn(self.input_dim, self.active_dim)
            W = orth(W)
        else:
            assert W.shape == (self.input_dim, self.active_dim)
        self.W = Param('W', W)
        self.W.set_prior(StiefelPrior(*W.shape))
        self.link_parameters(self.W, self.inner_kernel)

    def _get_Z(self, X):
        return None if X is None else np.dot(X, self.W)

    def _get_Zs(self, X, X2):
        return self._get_Z(X), self._get_Z(X2)

    @Cache_this(limit=5, ignore_args=())
    def K(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        """
        Z, Z2 = self._get_Zs(X, X2)
        return self.inner_kernel.K(Z, Z2)

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        """
        Evaluate only the diagonal of the covariance matrix.
        """
        return self.inner_kernel.Kdiag(self._get_Z(X))
    
    def gradients_X(self, dL_dK, X, X2=None):
        Z, Z2 = self._get_Zs(X, X2)
        tmp = self.inner_kernel.gradients_X(dL_dK, Z, Z2)
        return np.einsum('ik,jk->ij', tmp, self.W)

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        assert X2 is None
        Z = self._get_Z(X)
        self.inner_kernel.update_gradients_full(dL_dK, Z)
        dL_dZ = self.inner_kernel.gradients_X(dL_dK, Z)
        self.W.gradient = np.einsum('ij,ik->kj', dL_dZ, X)

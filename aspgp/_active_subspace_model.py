"""
An extension to GPRegression for actives subspaces.

Author:
    Ilias Bilionis
    Predictive Science Laboratory,
    School of Mechanical Engineering,
    Purdue University,
    West Lafayette, IN, USA

Date:
    12/27/2014

"""


from . import optimize_stiefel_seq
from . import optimize_stiefel
from . import ActiveSubspaceKernel
from . import ParallelizedGPRegression
from . import hmc_step
from . import hmc_step_stiefel
import math
import numpy as np
from collections import Iterable
from GPy.models import GPRegression
from GPy.kern import RBF
from pysmc import MCMCWrapper
from pdb import set_trace as keyboard

__all__ = ['ActiveSubspaceGPRegression']


global count
count = 0
def _W_obj_fun(W, obj):
    """
    The objective function used for the optimization with respect to ``W``.

    :param W:   The projection matrix.
    :parma obj: An :class:`aspgp.ActiveSubspaceGPRegression` object.
    """
    W_prev = obj.kern.W.view()
    obj.kern.W = W
    F = obj.objective_function()
    G = -obj.kern.W.gradient
    obj.kern.W = W_prev
    return F, G


def _W_log_pi(W, obj):
    F, G = _W_obj_fun(W, obj)
    return -F, -G

def _other_log_pi(x, obj):
    F, G = obj._objective_grads(x)
    return F, G


class ActiveSubspaceGPRegression(ParallelizedGPRegression):

    """
    An extension to GPRegression for actives subspaces.

    """
    x = []   
    y = []
    l = []

    def __init__(self, X, Y, inner_kernel, W=None, **kwargs):
        """
        Initialize the object.
        """
        kernel = ActiveSubspaceKernel(X.shape[1], inner_kernel, W=W)
        kernel.W.fix()
        super(ActiveSubspaceGPRegression, self).__init__(X, Y, kernel,
                                                         **kwargs)

    def _optimize_W(self, stiefel_options):
        """
        Optimize the object with respect to W.
        """
        #res = optimize_stiefel_seq(_W_obj_fun, self.kern.W, args=(self,),
        #                           **stiefel_options)
        res = optimize_stiefel(_W_obj_fun, self.kern.W, args=(self,),
                                   **stiefel_options)
        self.kern.W = res.X

    def _sample_W(self, iter=10, disp=False, **kwargs):
        """
        Sample W keeping the hyper-parameters constant.
        """
        count = 0
        for i in xrange(iter):
            X, a, log_p = hmc_step_stiefel(np.array(self.kern.W), _W_log_pi, args=(self,),
                                           **kwargs)
            self.kern.W = X
            count += a
            if disp:
                print '{0:d}\t: ar={1:1.2f}, log_p={2:1.6e}'.format(i + 1,
                                                                    float(count) / (i + 1),
                                                                    log_p)
        return count, log_p

    def _sample_other(self, iter=10, disp=False, **kwargs):
        """
        Sample all the other parameters.
        """
        count = 0
        for i in xrange(iter):
            x, a, log_p = hmc_step(self.optimizer_array, _other_log_pi, args=(self,),
                                   **kwargs)
            self.optimizer_array = x
            count += a
            if disp:
                print '{0:d}\t: ar={1:1.2f}, log_p={2:1.6e}'.format(i + 1,
                                                                    float(count) / (i + 1),
                                                                    log_p)
        return count, log_p

    def sample(self, iter=10, disp=False,
            W_opts={'iter': 5, 'disp': False, 'epsilon': 0.1, 'T': 50},
            other_opts={'iter': 5, 'disp': False, 'epsilon': 0.1, 'T': 10}):
        count_other = 0
        count_W = 0
        for i in xrange(iter):
            count_other, _ = self._sample_other(**other_opts)
            count_W, log_p = self._sample_W(**W_opts)
            if disp:
                print '{0:d}\t: ar_other={1:1.2f}, ar_W={2:1.2f}, log_p={3:1.6e}'.format(i + 1,
                                            float(count_other) / other_opts['iter'],
                                            float(count_W) / W_opts['iter'],
                                                                    log_p)
                                                    

    def _optimize_other(self, **kwargs):
        """
        Optimize with respect to all the other parameters.

        The options are the same as those of the classic ``GPRegression.optimize()``.
        """
        self.kern.W.fix()
        super(ActiveSubspaceGPRegression, self).optimize(**kwargs)
        self.kern.W.unconstrain()

    def optimize(self, max_it=1000, tol=1e-4, disp=False, stiefel_options={}, **kwargs):
        """
        Optimize the model.
        """
        if not isinstance(self.kern, ActiveSubspaceKernel):
            # Then just do whatever the super class does
            super(ActiveSubspaceGPRegression, self).optimize(optimizer=optimizer,
                                                             start=start,
                                                             **kwargs)
            return
        nit = 0
        if disp:
            print '{0:4s} {1:11s} {2:5s}'.format('It', 'F', '(F - F_old) / F_old')
            print '-' * 30

        old_optimization_runs = self.optimization_runs
        self.optimization_runs = []
        while nit <= max_it:
            self.x.append(nit) #Store iteration number history
            try:
                self.l.append(np.float(self.kern.inner_kernel.lengthscale.view()))
            except:
                self.l.append(np.asarray(self.kern.inner_kernel.lengthscale.view()))
            nit += 1
            old_log_like = -self.objective_function()
            self.y.append(old_log_like) #Store log likelihood history 
            self._optimize_W(stiefel_options)
            self._optimize_other(**kwargs)
            log_like = -self.objective_function()
            delta_log_like = (log_like - old_log_like) / math.fabs(old_log_like)
            if disp:
                print '{0:4s} {1:4.5f} {2:5e}'.format(
                        str(nit).zfill(4), log_like, delta_log_like)
            if delta_log_like < tol:
                if disp:
                    print '*** Converged ***'
                break
        class Foo(object):
            f_opt = None
            x_opt = None
        l_opt = Foo() #self.optimization_runs[-1]
        l_opt.f_opt = -log_like# self.objective_function()
        l_opt.x_opt = self.optimizer_array.copy()
        self.optimization_runs = old_optimization_runs + [l_opt]

    def get_active_model(self):
        """
        Get the active model.
        """
        Z = np.dot(self.X, self.kern.W)
        m = GPRegression(Z, self.Y, self.kern.inner_kernel)
        m.Gaussian_noise.variance = self.Gaussian_noise.variance
        return m

    def bic(self):
        """
        Return the bic score of the model.
        """
        return self.log_likelihood() - 0.5 * self.num_params * math.log(self.X.shape[0])

    @staticmethod
    def fit(X, Y, active_dim=1, return_all=False, **kwargs):
        """
        Fit the model and search for the dimensionality of the active subspace
        using the Bayesian information criterion (BIC).

        :param X:           The inputs.
        :param Y:           The outputs.
        :param active_dim:  A list we will search for the best dimension.
        :kwargs:            Keyword arguments for optimize_restarts.
        """
        if not isinstance(active_dim, Iterable):
            active_dim = [active_dim]
        for ad in active_dim:
            assert isinstance(ad, int) and ad > 0
        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        num_input = X.shape[1]
        num_samples = X.shape[0]
        models = []
        bics = []
        likes = []
        for ad in active_dim:
            k0 = RBF(ad, ARD=True)
            m = ActiveSubspaceGPRegression(X, Y, k0)
            m.optimize_restarts(**kwargs)
            like = m.log_likelihood()
            bic = like - 0.5 * m.num_params * math.log(num_samples)
            bics.append(bic)
            models.append(m)
            likes.append(like)
        like = np.array(like)
        bic = np.array(bic)
        best_model = models[np.argmax(bics)]
        if return_all:
            return best_model, np.array(active_dim), bics, likes, models
        else:
            return best_model, np.max(bics), np.max(likes)

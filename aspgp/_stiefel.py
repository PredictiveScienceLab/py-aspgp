"""
Optimization over a Stiefel manifold.

References:
    + Wen, Z. W., & Yin, W. T. (2013). A feasible method for optimization with
      orthogonality constraints. Mathematical Programming, 142(1-2), 397-434.
      doi: 10.1007/s10107-012-0584-1


Author:
    Rohit Tripathy, Ilias Bilionis
    Predictive Science Laboratory,
    School of Mechanical Engineering,
    Purdue University,
    West Lafayette, IN, USA

Date:
    12/27/2014

"""

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
try:
    from scipy.optimize import OptimizeResult
except:
    class OptimizeResult(object):
        pass
import math
import pybgo


class LSFunc(object):
        def __call__(self, tau):
            return self.func(Y_func(np.exp(tau[0]) * self.tau_max, self.X, self.A), *self.func_args)[0]


__all__ = ['optimize_stiefel', 'optimize_stiefel_seq']


# Line search methods available through minimize_scalar
LINE_SEARCH_METHODS = ['BRENT', 'BOUNDED', 'GOLDEN']


def compute_A(G, X):
    """
    :param G: Gradient of the function to be minimized
    :param X: A point on the Stiefel Manifold
    """
    g_trans = np.transpose(G)
    x_trans = np.transpose(X)
    A = np.dot(G, x_trans) - np.dot(X, g_trans)
    return A


def Y_func(tau, X, A):
    """
    :param tau: Step size
    :param X: Point on the Stiefel Manifold
    :param A: A matrix
    """
    I = np.eye(A.shape[0])
    B = I + 0.5 * tau * A
    C = I - 0.5 * tau * A
    Q = np.linalg.solve(B, C)
    return np.dot(Q, X)


def func_obj_seq(X, dmax, func, *args):
    """
    An objective function used by ``optimize_stiefel_seq``.
    """
    n = X.shape[0]
    d = X.shape[1]
    F, G = func(X, *args)
    return F, G[:, :d]


def func_obj_seq_w_zeros(X, dmax, func, *args):
    """
    An objective function used by ``optimize_stiefel_seq``.
    """
    n = X.shape[0]
    d = X.shape[1]
    F, G = func(np.hstack([X, np.zeros((n, dmax - d))]), *args)
    return F, G[:, :d]


def optimize_stiefel_seq(func, X0, args=(), tau_max=0.5, max_it=1, tol=1e-6,
                         disp=False, obj_func=func_obj_seq_w_zeros, **kwargs):
    """
    Optimize stiefel sequentially.
    """
    dmax = X0.shape[1]
    res = optimize_stiefel(obj_func, X0[:, :1], args=(dmax,func) + args, tau_max=tau_max,
                           max_it=max_it, tol=tol, disp=disp, **kwargs)
    for d in xrange(2, dmax + 1):
        Xd0 = res.X
        xr = X0[:, d-1:d]
        A = np.hstack([Xd0, xr])
        Q, R = np.linalg.qr(A)
        xr = Q[:, -1:]
        Xd0 = np.hstack([Xd0, xr])
        res = optimize_stiefel(obj_func, Xd0, args=(dmax,func) + args, tau_max=tau_max,
                               max_it=max_it, tol=tol, disp=disp, **kwargs)
    return res


def optimize_stiefel(func, X0, args=(), tau_max=.5, max_it=1, tol=1e-6,
                     disp=False, tau_find_freq=100):
    """
    Optimize a function over a Stiefel manifold.

    :param func: Function to be optimized
    :param X0: Initial point for line search
    :param tau_max: Maximum step size
    :param max_it: Maximum number of iterations
    :param tol: Tolerance criteria to terminate line search
    :param disp: Choose whether to display output
    :param args: Extra arguments passed to the function
    """
    tol = float(tol)
    assert tol > 0, 'Tolerance must be positive'
    max_it = int(max_it)
    assert max_it > 0, 'The maximum number of iterations must be a positive '\
                       + 'integer'
    tau_max = float(tau_max)
    assert tau_max > 0, 'The parameter `tau_max` must be positive.'
    k = 0
    X = X0.copy()
    nit = 0
    nfev = 0
    success = False
    if disp:
        print 'Stiefel Optimization'.center(80)
        print '{0:4s} {1:11s} {2:5s}'.format('It', 'F', '(F - F_old) / F_old')
        print '-' * 30

    
    ls_func = LSFunc()
    ls_func.func = func
    decrease_tau = False
    tau_max0 = tau_max
    while nit <= max_it:
        nit += 1
        F, G = func(X, *args)
        F_old = F
        nfev += 1
        A = compute_A(G, X)
        ls_func.A = A
        ls_func.X = X
        ls_func.func_args = args
        ls_func.tau_max = tau_max
        increased_tau = False
        if nit == 1 or decrease_tau or nit % tau_find_freq == 0:
            # Need to minimize ls_func with respect to each argument
            tau_init = np.linspace(-10, 0., 3)[:, None]
            tau_d = np.linspace(-10, 0., 50)[:, None]
            tau_all, F_all = pybgo.minimize(ls_func, tau_init, tau_d, fixed_noise=1e-16,
                    add_at_least=1, tol=1e-2, scale=True,
                    train_every=1)[:2]
            nfev += tau_all.shape[0]
            idx = np.argmin(F_all)
            tau = np.exp(tau_all[idx, 0]) * tau_max
            if tau_max - tau <= 1e-6:
                tau_max = 1.2 * tau_max
                if disp:
                    print 'increasing tau_max to {0:1.5e}'.format(tau_max)
                    increased_tau = True
            if decrease_tau:
                tau_max = .8 * tau_max
                if disp:
                    print 'decreasing max_tau to {0:1.5e}'.format(tau_max)
                decrease_tau = False
            F = F_all[idx, 0]
        else:
            F = ls_func([np.log(tau /  tau_max)])
        delta_F = (F_old - F) / np.abs(F_old)
        if delta_F < 0:
            if disp:
                print '*** backtracking'
            nit -= 1
            decrease_tau = True
            continue
        X_old = X
        X = Y_func(tau, X, A)
        if disp:
            print '{0:4s} {1:1.5e} {2:5e} tau = {3:1.3e}, tau_max = {4:1.3e}'.format(
             str(nit).zfill(4), F, delta_F, tau, tau_max)
        if delta_F <= tol:
            if disp:
                print '*** Converged ***'
            success = True
            break
    res = OptimizeResult()
    res.X = X
    res.nfev = nfev
    res.nit = nit
    res.fun = F
    res.success = success
    return res

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


#def ls_func(tau, func, func_args, X, A):
#    return func(Y_func(tau, X, A), *func_args)[0]


def optimize_stiefel_seq(func, X0, args=(), tau_max=0.1, max_it=100, tol=1e-3,
                         disp=False):
    """
    Optimize stiefel sequentially.
    """
    dmax = X0.shape[1]
    def func_tmp(X, dmax, *args):
        n = X.shape[0]
        d = X.shape[1]
        F, G = func(X, *args)
        return F, G[:, :d]
    res = optimize_stiefel(func_tmp, X0[:, :1], args=(dmax,) + args, tau_max=tau_max,
                           max_it=max_it, tol=tol, disp=disp)
    for d in xrange(2, dmax + 1):
        Xd0 = res.X
        xr = np.random.randn(Xd0.shape[0], 1)
        xr /= np.linalg.norm(xr)
        A = np.hstack([Xd0, xr])
        Q, R = np.linalg.qr(A)
        xr = Q[:, -1:]
        X0 = np.hstack([Xd0, xr])
        res = optimize_stiefel(func_tmp, X0, args=(dmax,) + args, tau_max=tau_max,
                               max_it=max_it, tol=tol, disp=disp)
    return res


def optimize_stiefel(func, X0, args=(), tau_max=1., max_it=100, tol=1e-3,
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

    class LSFunc(object):
        def __call__(self, tau):
            return self.func(Y_func(tau[0], self.X, self.A), *self.func_args)[0]
    ls_func = LSFunc()
    ls_func.func = func
    find_tau = True
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
        if find_tau or nit % tau_find_freq == 1:
            # Need to minimize ls_func with respect to each argument
            tau_init = np.linspace(0, tau_max, 3)[:, None]
            tau_d = np.linspace(0, tau_max, 50)[:, None]
            tau_all, F_all = pybgo.minimize(ls_func, tau_init, tau_d, fixed_noise=1e-16,
                    add_at_least=1, tol=1e-3, scale=True,
                    train_every=1)[:2]
            nfev += tau_all.shape[0]
            idx = np.argmin(F_all)
            tau = tau_all[idx, 0]
            if tau_max - tau <= 1e-6:
                tau_max = 1.2 * tau_max
            if tau < tau_max0:
                tau_max = tau_max0
            F = F_all[idx, 0]
            find_tau = False
        else:
            F = ls_func([tau])
        X_old = X
        X = Y_func(tau, X, A)
        delta_F = (F_old - F) / F_old
        if delta_F < 0:
            if disp:
                print '*** backtracking'
            nit -= 1
            find_tau = True
            continue
        if disp:
            print '{0:4s} {1:4.5f} {2:5e} tau = {3:1.5f}'.format(
             str(nit).zfill(4), F, delta_F, tau)
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

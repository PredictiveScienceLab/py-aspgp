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


__all__ = ['optimize_stiefel']


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


def ls_func(tau, func, func_args, X, A):
    return func(Y_func(tau, X, A), *func_args)[0]


def optimize_stiefel(func, X0, args=(), tau_max=0.1, max_it=100, tol=1e-3,
                     disp=False, line_search_method='brent'):
    """
    Optimize a function over a Stiefel manifold.

    :param func: Function to be optimized
    :param X0: Initial point for line search
    :param tau_max: Maximum step size
    :param max_it: Maximum number of iterations
    :param tol: Tolerance criteria to terminate line search
    :param disp: Choose whether to display output
    :param line_search_method: Choose between 3 line search algorithms:
                                Brent/ Golden/ Bounded
    :param args: Extra arguments passed to the function
    """
    assert line_search_method.upper() in LINE_SEARCH_METHODS, \
           'The line search can be one of: ' + str(LINE_SEARCH_METHODS)
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
    while nit <= max_it:
        nit += 1
        F, G = func(X, *args)
        nfev += 1
        A = compute_A(G, X)
        ls_args = (func, args, X, A)
        ls_res = minimize_scalar(ls_func, args=ls_args,
                                 bounds=(0., tau_max),
                                 method=line_search_method)
        nfev += ls_res.nfev
        if ls_res.x < 0:
            print '***'
            tau = np.linspace(0, tau_max, 100)
            y = [ls_func(t, *ls_args) for t in tau]
            idx = np.argmin(y)
            ls_res.x = tau[idx]
            ls_res.fun = y[idx]
        tau = ls_res.x
        X_old = X
        F_old = F
        F = ls_res.fun
        X = Y_func(tau, X, A)
        delta_F = (F_old - F) / F_old
        if disp:
            print '{0:4s} {1:4.5f} {2:5e}'.format(
             str(nit).zfill(4), F, delta_F)
        if delta_F < tol:
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

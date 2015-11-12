"""
Hamiltonian Monte Carlo implemented as in Neal 2011.
"""

import numpy as np
import math


def hmc_step(x0, log_pi, args=(), epsilon=0.1, T=10):
    x = x0.copy()
    p0 = np.random.randn(*x0.shape)
    p = p0.copy()
    U0, grad_U0 = log_pi(x0, *args)
    H0 = U0 + .5 * np.sum(p0 ** 2)
    p -= .5 * epsilon * grad_U0
    for i in xrange(T):
        x += epsilon * p
        U, grad_U = log_pi(x, *args)
        if i != T - 1:
            p -= epsilon * grad_U
    p -= .5 * epsilon * grad_U
    p *= -1.
    H = U + .5 * np.sum(p ** 2)
    u = np.random.rand()
    if u < math.exp(-H + H0):
        return x, 1, U
    return x0, 0, U0

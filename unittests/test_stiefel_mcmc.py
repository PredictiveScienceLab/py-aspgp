import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
import math


def log_pi(X, A):
    F = -.5 * np.einsum('ji,jk,ki', X, A, X)
    G = -np.einsum('jk,ki->ji', A, X)
    return F, G


def orth_stiefel_project(X, U):
    tmp = np.dot(X.T, U)
    return U - .5 * np.dot(X, tmp + tmp.T)


def hmc_step_stiefel(X0, log_pi, args=(), epsilon=.3, T=500):
    """
    Hamiltonian Monte Carlo for Stiefel manifolds.
    """
    n, d = X0.shape
    U = np.random.randn(*X0.shape)
    tmp = np.dot(X0.T, U)
    U = orth_stiefel_project(X0, U)
    log_pi0, G0 = log_pi(X0, *args)
    H0 = log_pi0 + .5 * np.einsum('ij,ij', U, U)
    X1 = X0.copy()
    G1 = G0
    for tau in xrange(T):
        U += 0.5 * epsilon * G1
        U = orth_stiefel_project(X0, U)
        A = np.dot(X1.T, U)
        S = np.dot(U.T, U)
        exptA = scipy.linalg.expm(-epsilon * A)
        tmp0 = np.bmat([X0, U])
        tmp1 = scipy.linalg.expm(epsilon * np.bmat([[A, -S], 
                                                     [np.eye(d), A]]))
        tmp2 = scipy.linalg.block_diag(exptA, exptA)
        tmp3 = np.dot(tmp0, np.dot(tmp1, tmp2))
        X1 = tmp3[:, :d]
        U = tmp3[:, d:]
        log_pi1, G1 = log_pi(X1, *args)
        U += 0.5 * epsilon * G1
        U = orth_stiefel_project(X0, U)
    H1 = log_pi1 + .5 * np.einsum('ij,ij', U, U)
    u = np.random.rand()
    if u < math.exp(-H1 + H0):
        return X1, 1, log_pi1
    return X0, 0, log_pi0


dim = 50
sdim = 2
A = scipy.linalg.toeplitz([2, -1] + [0] * (dim-2))
w, v = scipy.linalg.eigh(A)

X0 = np.random.randn(dim, sdim)
X0 = scipy.linalg.orth(X0)

X = X0
count = 0
l = []
for i in xrange(100):
    X, a, log_p = hmc_step_stiefel(X, log_pi, args=(A,))
    count += a
    print '{0:d}\t: ar={1:1.2f}, log_p={2:1.6e}'.format(i + 1,
                                                        float(count) / (i + 1),
                                                        log_p)
    l.append(log_p)
print np.dot(X.T, X)
print np.dot(v[:, :sdim].T, v[:, :sdim])
print np.linalg.norm(np.dot(A, X), axis=0) / np.linalg.norm(X, axis=0)
print w[:sdim]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(l)
fig, ax = plt.subplots()
ax.plot(X, 'b')
ax.plot(v[:, :sdim], 'r')
plt.show()

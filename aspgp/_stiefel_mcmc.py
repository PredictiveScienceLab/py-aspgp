import numpy as np
import scipy.linalg
import math


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
    H0 = log_pi0 - .5 * np.einsum('ij,ij', U, U)
    X1 = X0.copy()
    G1 = G0
    for tau in xrange(T):
        X10 = X1.copy()
        S = np.dot(U.T, U)
        A = np.dot(X10.T, U)
        U += 0.5 * epsilon * G1
        U = orth_stiefel_project(X10, U)
        exptA = scipy.linalg.expm(-epsilon * A)
        tmp0 = np.bmat([X10, U])
        tmp1 = scipy.linalg.expm(epsilon * np.bmat([[A, -S], 
                                                     [np.eye(d), A]]))
        tmp2 = scipy.linalg.block_diag(exptA, exptA)
        tmp3 = np.dot(tmp0, np.dot(tmp1, tmp2))
        X1 = tmp3[:, :d]
        U = tmp3[:, d:]
        log_pi1, G1 = log_pi(X1, *args)
        U += 0.5 * epsilon * G1
        U = orth_stiefel_project(X10, U)
        X10 = X1.copy()
    H1 = log_pi1 - .5 * np.einsum('ij,ij', U, U)
    u = np.random.rand()
    if u < math.exp(H1 - H0):
        print 'accept'
        return X1, 1, log_pi1
    print 'reject'
    return X0, 0, log_pi0

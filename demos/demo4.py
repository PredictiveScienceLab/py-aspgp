"""
Testing with the data of Paul Constantine.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
import os
ASPGP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ASPGP_DIR)
import aspgp
import GPy
import pybgo
from mpi4py import MPI as mpi
rank = mpi.COMM_WORLD.Get_rank()


DEMO_DIR = os.path.join(ASPGP_DIR, 'demos')
PC_DIR = os.path.join(DEMO_DIR, 'gc_amp_10')
X_FILE = os.path.join(PC_DIR, 'X.npy')
Y_FILE = os.path.join(PC_DIR, 'Y.npy')

X = np.load(X_FILE)
Y = np.load(Y_FILE)

i = 10
n = 47
R = X[:, :n]
E = X[:, n:2*n]
v = X[:, -1][:, None]

Ri = R[:, i-1][:, None]
Ei = E[:, i-1][:, None]
Rl = R[:, -1][:, None]
El = E[:, -1][:, None]
R_rest = np.hstack([R[:, :i-1], R[:, i:]])
E_rest = np.hstack([E[:, :i-1], E[:, i:]])
X = np.hstack([R_rest, E_rest, Ri, Rl, Ei, El, v])

X_train = X[:400, :]
Y_train = Y[:400, :]
X_val = X[400:, :]
Y_val = Y[400:, :]

dim = 8

k = GPy.kern.Matern32(dim, ARD=True)
stiefel_opt = {'disp': False,
               'tau_max': .01,
               'tol': 1e-5,
               'tau_find_freq': 10,
               'max_it': 10000}

m = aspgp.ActiveSubspaceGPRegression(X_train, Y_train, k, fixed_cols=5)
m.likelihood.variance.constrain_fixed(0.)
m.optimize_restarts(10, stiefel_options=stiefel_opt, comm=mpi.COMM_WORLD)
#m.optimize(stiefel_options=stiefel_opt)

if rank == 0:
    print str(m)
    print m.kern.Mat32.lengthscale

    print 'Normalization test:'
    print np.dot(m.kern.W.T, m.kern.W)

    fig, ax = plt.subplots()
    W = np.array(m.kern.W)
    ax.plot(W, 'x', markersize=10, markeredgewidth=2)
    fig, ax = plt.subplots()
    Y_p, V_p = m.predict(X_val)
    ax.plot(Y_p, Y_val, '.')
    plt.show()

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


np.random.seed(12345 * (rank + 1))


DEMO_DIR = os.path.join(ASPGP_DIR, 'demos')
PC_DIR = os.path.join(DEMO_DIR, 'pc2_data')
X_FILE = os.path.join(PC_DIR, 'X.npy')
Y_FILE = os.path.join(PC_DIR, 'Y.npy')

X_f = np.load(X_FILE)
Y_f = np.load(Y_FILE)
X = X_f[:240, :]
X_v = X_f[240:, :]
#Y_f = (Y_f - np.mean(Y_f)) / np.std(Y_f)
Y = Y_f[:240, :]
Y_v = Y_f[240:, :]


dim = 2

k = GPy.kern.Matern32(dim, ARD=True)
#k.lengthscale.set_prior(GPy.priors.Jeffreys())
#k.variance.set_prior(GPy.priors.Jeffreys())
stiefel_opt = {'disp': False,
               'tau_max': 0.5,
               'tol': 1e-3,
               'tau_find_freq': 10,
               'max_it': 1}

m = aspgp.ActiveSubspaceGPRegression(X, Y, k)
#m.likelihood.variance.set_prior(GPy.priors.Jeffreys())
m.optimize_restarts(20, tol=1e-3, disp=False, stiefel_options=stiefel_opt,
                    comm=mpi.COMM_WORLD)

if rank == 0:
    fig, ax = plt.subplots()
    Y_p, V_p = m.predict(X_v)
    ax.plot(Y_p, Y_v, '.') 
    
    print str(m)
    print m.kern.Mat32.lengthscale

    print 'Normalization test:'
    print np.dot(m.kern.W.T, m.kern.W)

    #fig, ax = plt.subplots()
    #W = np.array(m.kern.W)
    #ax.plot(W, 'x', markersize=10, markeredgewidth=2)
    plt.show()

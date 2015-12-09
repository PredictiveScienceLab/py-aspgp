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


dim = 2

k = GPy.kern.Matern32(dim, ARD=True)#, lengthscale=[18.4966923967, 7000.], variance=4.57012948037e-05)
stiefel_opt = {'disp': False,
               'tau_max': .01,
               'tol': 1e-5,
               'tau_find_freq': 10,
               'max_it': 10000}

m = aspgp.ActiveSubspaceGPRegression(X, Y, k)
m.optimize_restarts(500, stiefel_options=stiefel_opt, comm=mpi.COMM_WORLD)
#m.sample(iter=50, disp=True)
#m.optimize(stiefel_options=stiefel_opt)
#print m.kern.W
if rank == 0:
    print str(m)
    print m.kern.Mat32.lengthscale

    print 'Normalization test:'
    print np.dot(m.kern.W.T, m.kern.W)

    fig, ax = plt.subplots()
    W = np.array(m.kern.W)
    ax.plot(W, 'x', markersize=10, markeredgewidth=2)
    plt.show()

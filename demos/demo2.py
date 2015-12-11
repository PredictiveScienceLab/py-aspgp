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
PC_DIR = os.path.join(DEMO_DIR, 'pc2_data')
X_FILE = os.path.join(PC_DIR, 'X.npy')
Y_FILE = os.path.join(PC_DIR, 'Y.npy')
G_FILE = os.path.join(PC_DIR, 'G.npy')

X = np.load(X_FILE)
Y = np.load(Y_FILE)
G = np.load(G_FILE)


dim = 1
ck = GPy.kern.Matern32(dim, ARD=True)
cm = aspgp.ClassicActiveSubspaceGPRegression(X, Y, G, ck)
cm.optimize(messages=True)
#print str(cm)
#print cm.kern.Mat32.lengthscale


k = GPy.kern.RBF(dim, ARD=True)#, lengthscale=[18.4966923967, 7000.], variance=4.57012948037e-05)
stiefel_opt = {'disp': False,
               'tau_max': 0.5,
               'tol': 1e-6,
               'tau_find_freq': 1,
               'max_it': 1}
np.random.seed(1234)
m = aspgp.ActiveSubspaceGPRegression(X, Y, k)
#m.sample(iter=50, disp=True)
#m.optimize_restarts(10, stiefel_options=stiefel_opt, disp=True)
m.optimize(tol=1e-6, stiefel_options=stiefel_opt, disp=True)
#print m.kern.W
if rank == 0:
    print str(m)
    print m.kern.inner_kernel.lengthscale

    print 'Compare to:'
    print str(cm)
    print cm.kern.inner_kernel.lengthscale

    print 'Normalization test:'
    print np.dot(m.kern.W.T, m.kern.W)

    fig, ax = plt.subplots()
    ax.plot(cm.kern.W, 'o', markersize=10, markeredgewidth=2)
    W = np.array(m.kern.W)
    if W[0, 0] * cm.kern.W[0, 0] < 0.:
        W *= -1.
    ax.plot(W, 'x', markersize=10, markeredgewidth=2)
    plt.show()

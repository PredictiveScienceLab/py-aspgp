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
PC_DIR = os.path.join(DEMO_DIR, 'dyn_data')
X_FILE = os.path.join(PC_DIR, 'X_train.npy')
Y_FILE = os.path.join(PC_DIR, 'Y_ts.npy')
Xv_FILE = os.path.join(PC_DIR, 'X_valid.npy')
Yv_FILE = os.path.join(PC_DIR, 'Y_vs.npy')

X = np.load(X_FILE)
Y = np.load(Y_FILE)
Xv = np.load(Xv_FILE)
Yv = np.load(Yv_FILE)


np.random.seed(1234)
dim = 2
k = GPy.kern.Matern32(dim, ARD=True)
stiefel_opt = {'disp': False,
               'tau_max': .5,
               'tol': 1e-3,
               'tau_find_freq': 10,
               'max_it': 1}
#stiefel_opt = {}

m = aspgp.ActiveSubspaceGPRegression(X, Y, k)
m.randomize()
m.optimize_restarts(num_restarts=10, tol=1e-3,
           maxiter=1, stiefel_options=stiefel_opt, disp=False)

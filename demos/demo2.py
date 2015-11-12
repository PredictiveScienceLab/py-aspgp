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


DEMO_DIR = os.path.join(ASPGP_DIR, 'demos')
PC_DIR = os.path.join(DEMO_DIR, 'pc1_data')
X_FILE = os.path.join(PC_DIR, 'X.npy')
Y_FILE = os.path.join(PC_DIR, 'Y.npy')
G_FILE = os.path.join(PC_DIR, 'G.npy')

X = np.load(X_FILE)
Y = np.load(Y_FILE)
G = np.load(G_FILE)


dim = 3
ck = GPy.kern.Matern32(dim, ARD=True)
cm = aspgp.ClassicActiveSubspaceGPRegression(X, Y, G, ck)
cm.optimize(messages=True)
print str(cm)
print cm.kern.Mat32.lengthscale


k = GPy.kern.Matern32(dim, ARD=True)#, lengthscale=[18.4966923967, 7000.], variance=4.57012948037e-05)
stiefel_opt = {'disp': False,
               'tau_max': 1.,
               'tol': 1e-5,
               'max_it': 10000}

m = aspgp.ActiveSubspaceGPRegression(X, Y, k)
m.sample(iter=1000, disp=True)
print str(m)
print m.kern.Mat32.lengthscale


fig, ax = plt.subplots()
ax.plot(cm.kern.W, 'o', markersize=10, markeredgewidth=2)
ax.plot(m.kern.W, 'x', markersize=10, markeredgewidth=2)
plt.show()

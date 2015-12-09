"""
Test the code that mixes active subspaces with common GP's.
"""

import aspgp
import GPy

# Test Stiefel prior
pW = aspgp.StiefelPrior(10, 4, fixed_cols=2)
W = pW.rvs(1)
print 'W'
print W.reshape((10, 4))
G = pW.lnpdf_grad(W)
print 'G'
print G.reshape((10, 4))

# Test the kernel
k0 = GPy.kern.Matern32(4, ARD=True)
k = aspgp.ActiveSubspaceKernel(10, k0, fixed_cols=2)

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
sys.path.insert(0, '.')
from aspgp import optimize_stiefel_seq
from aspgp import optimize_stiefel


def f(X, A):
    F = .5 * np.einsum('ji,jk,ki', X, A, X)
    G = np.einsum('jk,ki->ji', A, X)
    return F, G

dim = 10
sdim = 2
A = scipy.linalg.toeplitz([2, -1] + [0] * (dim-2))
w, v = scipy.linalg.eigh(A)

X0 = np.random.randn(dim, sdim)
X0 = X0 / np.linalg.norm(X0, axis=0)

res = optimize_stiefel_seq(f, X0, args=(A,), tau_max=1., max_it=10000, tol=1e-20,disp=True)
X = res.X
print np.linalg.norm(np.dot(A, X), axis=0) / np.linalg.norm(X, axis=0)
print w[:sdim]

plt.plot(X, 'b')
plt.plot(v[:, :sdim], 'r')
plt.show()

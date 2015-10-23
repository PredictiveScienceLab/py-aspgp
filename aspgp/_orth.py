"""
Orthogonalization routines.

Author:
    Ilias Bilionis

Date:
    3/31/2015

"""


__all__ = ['append_orth_column']


import numpy as np


def append_orth_column(W, w):
    """
    Appends an orthogonal column to ``W`` so that the new matrix
    spans the same subspace as ``[W, w]``. It assumes that ``W``
    is already an orthogonal matrix and it does not change its contents.
    """
    proj = np.dot(W.T, w).flatten()
    w_new = w - np.sum(W * proj, axis=1)[:, None]
    w_new /= np.linalg.norm(w_new)
    W_new = np.hstack([W, w_new])
    return W_new

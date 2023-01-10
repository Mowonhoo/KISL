# -*- coding: UTF-8 -*-
from scipy.spatial.distance import pdist, squareform
import numpy as np
 
from numba import jit, float32


def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor  


def my_dcorr(X):
    """ 
    ### Compute the distance correlation function ###
    from dcor import distance_correlation as dcorr
    input: pd.DataFrame.object (n*m)
    >>> a = nunpy.array
    >>> b = nunpy.array
    >>> my_dcorr(a, b)
    value: pd.DataFrame.object (n*n)
    """ 
    dcor = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for i in range(X.shape[0]):#enumerate
        for j in range(i, X.shape[0]):
            dcor[i][j] = dcor[j][i] = dcorr(np.array(X.iloc[i]), np.array(X.iloc[j]))
    dcor = pd.DataFrame(dcor, index=X.index, columns=X.index)        
    return dcor


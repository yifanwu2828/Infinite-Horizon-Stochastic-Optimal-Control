import numpy as np
from numba import jit
from icecream import ic


sigma = np.array([0.04, 0.04, 0.004])
cov = np.diag(sigma**2)

log_2pi = np.log(2*np.pi)
# `eigh` assumes the matrix is Hermitian.
vals, vecs = np.linalg.eigh(cov)
logdet = np.sum(np.log(vals))
valsinv = np.array([1. / v for v in vals])
# `vecs` is R times D while `vals` is a R-vector where R is the matrix
U = vecs * np.sqrt(valsinv)
rank = len(vals)


@jit(nopython=True, cache=True, fastmath=True)
def logpdf(x, mean):
    dev = x - mean
    # "maha" for "Mahalanobis distance".
    maha = np.square(np.dot(dev, U)).sum(axis=-1)
    log_prob = -0.5 * (rank * log_2pi + maha + logdet)
    return log_prob


@jit(nopython=True, cache=True, fastmath=True)
def logsumexp(x):
    c = x.max()
    ans = c + np.log(np.sum(np.exp(x - c)))
    return ans


@jit(nopython=True, cache=True, fastmath=True)
def softmax(x):
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x))

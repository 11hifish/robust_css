import numpy as np


def rank_k_svd(A, k):
    d, n = A.shape
    true_max_rank = min(d, n)
    k = min(k, true_max_rank)
    u, sigma, vh = np.linalg.svd(A)
    u = u[:, :k]
    sigma = np.diag(sigma[:k])
    vh = vh[:k]
    return u @ sigma @ vh

import numpy as np


def generate_synthetic_matrix(rank, n):
    """
        Returns a (k + n) x (k + n) matrix.
    :param rank: target rank -- k
    :param n: matrix size parameter
    :return: a synthetic matrix
    """
    M = np.zeros((rank + n, rank + n))
    M[:rank, :rank] = np.eye(rank) * (n ** (3 / 2))
    M[rank:, rank:] = np.ones((n, n))
    return M

def test():
    rank = 2
    n = 3
    M = generate_synthetic_matrix(rank, n)
    print(M)

if __name__ == '__main__':
    test()

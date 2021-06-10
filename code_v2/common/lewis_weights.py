"""
lewis_weights.py - This file contains the following:
- l1 Lewis weight sampling
- OSNAP matrices, used for approximate l2 leverage scores
- Some code to test Lewis weight sampling

- approximate_leverage_scores might be further sped up
by speeding up the computation of SA, where S is an
OSNAP matrix.
"""
import numpy as np

########################################################


def generate_OSNAP_sparse_embedding(n_row, n_col, s):
    emb = np.zeros((n_row, n_col))
    for c in range(n_col):
        sel_idx = np.random.choice(np.arange(n_row), s, replace=False)
        sign = np.random.randint(2, size=s) * 2 - 1
        emb[sel_idx, c] = sign
    return emb * 1 / np.sqrt(s)


def approximate_leverage_scores(A):
    n, d = A.shape

    # 1. Generate a Countsketch Matrix
    num_rows = 5 * d
    S = generate_OSNAP_sparse_embedding(n_row=num_rows, n_col=n, s=1)

    # 2. Take QR decomposition
    SA = S @ A
    _, R_inv = np.linalg.qr(SA)
    R = np.linalg.pinv(R_inv)

    # 3. Multiply by a JL matrix
    num_JL_cols = int(5 * np.log(n))
    G = 1/np.sqrt(num_JL_cols) * np.random.normal(size=(d, num_JL_cols))
    RG = R @ G
    well_conditioned_basis = A @ RG

    # 4. Compute leverage scores
    return np.linalg.norm(well_conditioned_basis, axis=1) ** 2


def lewis_iterate(A, p, w):
    w_to_exp = np.power(w, 1/2 - 1/p, where=w!=0, out=np.zeros_like(w))
    w_to_exp = np.expand_dims(w_to_exp, axis=-1)
    approx_leverage_scores = approximate_leverage_scores(w_to_exp * A)
    new_weights = ((w**(2/p - 1)) * approx_leverage_scores)**(p/2)
    return np.reshape(new_weights, w.shape)


def lewis_iterate_exact(A, p, w):
    w_to_exp = np.power(w, 1 - 2 / p, where=w != 0, out=np.zeros_like(w))
    M = A.T @ np.diag(w_to_exp) @ A
    M_inv = np.linalg.pinv(M)
    Res = A @ M_inv @ A.T
    w = np.diagonal(Res) ** (p / 2)
    return w


def approximate_lewis_weights(A, p, T, exact=False, verbose=False):
    n = A.shape[0]
    w = np.ones(n)
    for t in range(T):
        if verbose:
            print('Lewis iterate {}/{}'.format(t, T))
        if exact:
            w = lewis_iterate_exact(A, p, w)
        else:
            w = lewis_iterate(A, p, w)
    return w


def l1_lewis_weights(A, approx_factor=100, exact=False):
    """
    Row wise Lewis weights for p = 1.
    :param A: data matrix, size n x d
    :return: row wise Lewis weight, size n
    """

    # First ensure that A has full rank.
    rank = np.linalg.matrix_rank(A)
    U, _, _ = np.linalg.svd(A)
    U = U[:, :rank]
    A = U

    # Compute Lewis weights for full rank basis.
    T = int(approx_factor * np.log(np.log(A.shape[0])))
    T = max(T, 10)
    return approximate_lewis_weights(A, 1, T, exact=exact)


def get_lewis_weight_sampling_matrix(A, lewis_weights, sample_rows, p):
    n, _ = A.shape
    assert(len(lewis_weights) == n)

    # Rescale lewis_weights so that they sum to N = sample_rows
    sampling_values = sample_rows * lewis_weights/np.sum(lewis_weights)
    probabilities = sampling_values/np.sum(sampling_values)
    sel_indices = np.random.choice(n, size=sample_rows, p=probabilities, replace=False)
    entries = 1 / (sampling_values[sel_indices] ** (1 / p))
    S = np.zeros((sample_rows, n))
    S[np.arange(sample_rows), sel_indices] = entries
    return S, sel_indices


def perform_l1_lewis_weight_sampling(A, sample_rows, approx_factor=25, exact=False):
    lewis_weights = l1_lewis_weights(A, approx_factor, exact=exact)
    S, sel_indices = get_lewis_weight_sampling_matrix(A, lewis_weights, sample_rows, p=1)
    return S, sel_indices

########################################################

## Check result
def calc_p_norm(A, p):
    return np.sum(np.abs(A) ** p) ** (1/p)


def check_result(A, B, p):
    _, d = A.shape
    rand_x = np.random.rand(d) * 100
    A_p_norm = calc_p_norm(np.matmul(A, rand_x), p)
    B_p_norm = calc_p_norm(np.matmul(B, rand_x), p)
    print('Ax norm {}'.format(A_p_norm))
    print('SAx norm {}'.format(B_p_norm))
    return A_p_norm / B_p_norm


if __name__ == "__main__":
    
    # Test approximate_leverage_scores
    print("=======================================================")
    print("Testing Approximate Leverage Scores")
    A = np.random.normal(size=(1000, 30))
    # A = np.array([[1, 2, 3], [5, 0, 0], [5, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
    Q, _ = np.linalg.qr(A)
    true_leverage_scores = np.linalg.norm(Q, axis=1)
    approx_leverage_scores = approximate_leverage_scores(A)
    approx_max = np.max(approx_leverage_scores/true_leverage_scores)
    approx_min = np.min(approx_leverage_scores/true_leverage_scores)
    print("Approximation Factors: ", approx_max, approx_min)

    # Test Lewis weights
    print("=======================================================")
    print("Testing Lewis weights approximation factor:")
    n, d = 1000, 100
    A = np.random.rand(n, d) + 0.001
    A = A * 1000
    S, _ = perform_l1_lewis_weight_sampling(A, 1000)
    x = np.random.rand(d) * 1000
    approx_factor = check_result(A, S @ A, 1)
    print('approx factor Ax / SAx {}'.format(approx_factor))

    print("=======================================================")
    print("Testing Lewis weights values")
    print("2nd and 3rd Lewis weights should be 5 times the last three")
    M_test = [[1, 2, 3], [5, 0, 0], [5, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
    M_test = np.array(M_test)
    print(l1_lewis_weights(M_test))


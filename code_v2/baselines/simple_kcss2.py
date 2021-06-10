import numpy as np
from tt.maxvol import maxvol
from code_v2.baselines.rank_k_svd import rank_k_svd


def simple_kcss2(A, rank):
    """
    k columns, s.t. <= (k sqrt(log k)) |A - A_k|_F
    :param A: size d x n
    :param rank: rank k
    :return: k columns of A
    """
    # stage 1: initial setup
    _, n = A.shape
    _, _, vh = np.linalg.svd(A)
    V_k = vh[:rank].T  # size n x k
    part1 = np.linalg.norm(V_k, axis=1) ** 2 / (2 * rank)  # size n
    AVV_T = A @ V_k @ V_k.T
    part2_top = np.linalg.norm(A, axis=0) ** 2 - np.linalg.norm(AVV_T, axis=0) ** 2  # size n
    part2_bottom = 2 * (np.linalg.norm(A, ord='fro') ** 2 - np.linalg.norm(AVV_T, ord='fro') ** 2)
    part2 = part2_top / part2_bottom
    sampling = part1 + part2  # size n
    # stage 2: randomized phase
    c = int(rank * np.log(rank))
    if c > n:
        sel_idx = np.arange(n)
        A_sel = A
        T = V_k.T
    else:
        probabilities = sampling / np.sum(sampling)
        sel_idx = np.random.choice(n, size=c, p=probabilities, replace=False)
        entries = 1 / np.sqrt(sampling[sel_idx])
        SD = np.zeros((n, c))
        SD[sel_idx, np.arange(c)] = entries
        A_sel = A[:, sel_idx]  # d x c
        T = V_k.T @ SD  # k x c
    # stage 3: deterministic phase
    fix_rank_sel_idx = maxvol(T.T, rank)  # c x k => k x k
    cols = A_sel[:, fix_rank_sel_idx]
    final_sel_idx = sel_idx[fix_rank_sel_idx]
    return cols, final_sel_idx


def compare_Fro_norm_squared(A, cols, rank):
    A_k = rank_k_svd(A, rank)
    l2_error_svd = np.linalg.norm(A - A_k, ord='fro') ** 2
    q, r = np.linalg.qr(cols)
    l2_error_css = np.linalg.norm(A - q @ q.T @ A, ord='fro') ** 2
    print('l2 error svd: {}'.format(l2_error_svd))
    print('l2 error css: {}'.format(l2_error_css))
    print('ratio: {}'.format(l2_error_css / l2_error_svd))


def test_simple_kcss2():
    # A = np.array([[1,2,3,4,8,10],
    #               [5,6,7,8,3,5],
    #               [9,0,0,2,6,0]])
    # rank = 3
    A = np.random.normal(0, 10, size=(50, 100))
    rank = 10
    cols, sel_idx = simple_kcss2(A, rank)
    # print('cols')
    # print(cols)
    # print('final sel idx', sel_idx)
    compare_Fro_norm_squared(A, cols, rank)


if __name__ == '__main__':
    test_simple_kcss2()


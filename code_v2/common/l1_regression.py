"""
l1_regression.py - This file contains the following:
- Fast l1 regression: an implementation of pwSGD
- Our previous l1 regression algorithm: a slower baseline using MOSEK

"""
import numpy as np
import cvxpy as cp
import ray

# Parameters
MAXITER_L1 = 3000 # maximum iterations of pwSGD

########################################################################
# Slow l1 regression - Baseline using MOSEK

"""
solve_l1_regression_MOSEK - MOSEK-based l1
regression solver on a single column.
"""
@ray.remote
def solve_l1_regression_MOSEK(A, b, c_idx=None):
    # print('start solving l1 regression for column {}'.format(c_idx))
    m, n = A.shape
    b = b.ravel() 
    x = cp.Variable(n) 
    t = cp.Variable(m) 

    # objective
    objective = cp.sum(t) 

    # contraints 
    constraints = [cp.matmul(A, x) - b <= t, -cp.matmul(A, x) + b <= t, t >= 0]

    # problem 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
    except:
        print("MOSEK FAILED")
        problem.solve(solver=cp.GLPK, verbose=False)

    if problem.status in ["infeasible", "unbounded"]:
        print('Problem status: {}'.format(problem.status))
        return None
    else:
        # print('c_idx: {}, sol: {}'.format(c_idx, x.value))
        return np.sum(np.abs(A @ x.value - b))


def compute_l1_error(U, A):
    """
    Compute the error: min_v |UV - A|_1
    :param U: Selected columns (left factor).
    :param A: The original data matrix of size d x n.
    :param verbose: debugging mode.
    :return: l1 error.
    """
    _, n = A.shape

    total_error = []
    for i in range(n):
        col = A[:, i].ravel()
        res = solve_l1_regression_MOSEK.remote(U, col)
        if res is None:
            res = 0
        total_error.append(res)
    results = ray.get(total_error)
    results = np.sum(np.array(results))
    return results

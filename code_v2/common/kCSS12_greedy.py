import numpy as np 
import math

DELTA_DEFAULT = 0.000001
MIN_TOLERANCE = 1e-9

def norm_l12(A): 
	return sum(np.linalg.norm(A, ord=2, axis=0))

def find_V_l12(U, A): 
	V = np.linalg.pinv(U) @ A
	return V 

# Target_k is the desired number of columns selected
# It is needed when using the lazier_than_lazy technique.
def pick_first_column_greedy_l12(A, target_k, lazier_than_lazy=True, delta=DELTA_DEFAULT): 
	num_cols = A.shape[-1] 
	best_col_idx = -1 
	min_cost = float("inf")

	if lazier_than_lazy:
		num_sampled_cols = math.ceil(num_cols * np.log(1/delta) / target_k)
		num_sampled_cols = min(num_cols, num_sampled_cols)
		test_col_indices = np.random.choice(num_cols, size=num_sampled_cols, replace=False)
	else:
		test_col_indices = np.arange(num_cols)

	for i in test_col_indices: 
		current_column = np.expand_dims(A[:, i], axis=-1)
		v_row = find_V_l12(current_column, A) 
		current_cost = norm_l12(A - current_column @ v_row)
		if current_cost <= min_cost + MIN_TOLERANCE: 
			min_cost = current_cost
			best_col_idx = i

	if best_col_idx == -1:
		best_col_idx = np.random.randint(num_cols)

	return best_col_idx 

# Target_k is the desired number of columns selected.
# It is needed when using the lazier_than_lazy technique.
# (Set to -1 when not using the lazier_than_lazy technique.)
def pick_new_column_greedy_l12(A, U_old, V_old, target_k, selected_indices, lazier_than_lazy=True, delta=DELTA_DEFAULT):
	num_rows, num_cols = A.shape 
	min_cost = norm_l12(A - U_old @ V_old)
	best_col_idx = -1 
	V_new = None

	zero_column = np.zeros((num_rows, 1))
	U_new = np.concatenate([U_old, zero_column], axis=-1)

	if lazier_than_lazy:
		# Note that now, we have to exclude the indices of the columns 
		# already selected in U, when sampling columns (this is done
		# through the probability vector p below).
		num_sampled_cols = int(math.ceil(num_cols * np.log(1/delta) / target_k))
		p = np.ones(num_cols)
		p[selected_indices] = 0
		num_cols_left = int(np.sum(p))
		p = p/np.sum(p)
		num_sampled_cols = min(num_cols_left, num_sampled_cols)
		test_col_indices = np.random.choice(np.arange(num_cols), size=num_sampled_cols, replace=False, p=p)
	else:
		test_col_indices = np.arange(num_cols)	

	for k in test_col_indices: 
		U_new[:, -1] = A[:, k]
		V_test = find_V_l12(U_new, A)
		new_cost = norm_l12(A - U_new @ V_test) 
		if new_cost <= min_cost + MIN_TOLERANCE: 
			min_cost = new_cost
			best_col_idx = k
			V_new = V_test 
	
	if best_col_idx == -1:
		best_col_idx = np.random.randint(num_cols)
		U_new[:, -1] = A[:, best_col_idx]
		V_new = find_V_l12(U_new, A)
	else:
		U_new[:, -1] = A[:, best_col_idx]
	return U_new, V_new, best_col_idx 

# Note that the runtime dependence on delta is log(1/delta), so
# we can set it to be small, and it still won't really affect
# the time complexity. (In each round the exact number of columns
# sampled is n * ln(1/delta)/k.)
def greedy_CSS_l12(A, num_cols, lazier_than_lazy=True):
	indices = []
	first_column_idx = pick_first_column_greedy_l12(A, target_k=num_cols)
	indices.append(first_column_idx) 
	U_greedy = np.expand_dims(A[:, first_column_idx], axis=-1)
	V_greedy = find_V_l12(U_greedy, A) 
	for k in range(num_cols-1): 
		U_greedy, V_greedy, idx = \
			pick_new_column_greedy_l12(A, U_greedy, V_greedy,
									   lazier_than_lazy=lazier_than_lazy,
									   target_k=num_cols, selected_indices=indices)
		indices.append(idx)
	return U_greedy, V_greedy, indices

if __name__ == '__main__':
	A = np.random.normal(size=(500, 500))
	U, V, idx = greedy_CSS_l12(A, 60)
	print(norm_l12(U @ V - A))
	U_test = A[:, idx]
	print(norm_l12(U - U_test))

	U, V, idx = greedy_CSS_l12(A, 60, lazier_than_lazy=False)
	print(norm_l12(U @ V - A))

import ray
ray.init(address='auto', _redis_password='5241590000000000')
from code_v2.common.utils import load_dataset, even_split_data, generate_dense_cauchy_matrix
from code_v2.common.kCSS12 import regular_CSS_l12
from code_v2.common.kCSS12_greedy import greedy_CSS_l12
from code_v2.common.l1_regression import *
from code_v2.common.generate_synthetic import generate_synthetic_matrix
from code_v2.common.lewis_weights import perform_l1_lewis_weight_sampling
import time
import argparse
import os
import pickle


parser = argparse.ArgumentParser(description='Distributed protocol.')
parser.add_argument('--data', type=str, default='techtc',
                    help='Dataset.')
parser.add_argument('--dataset-folder', type=str, default='/home/ubuntu/robust_css/dataset',
                    help='Dataset folder.')
parser.add_argument('--rank', type=int, default=100,
                    help='Rank.')
parser.add_argument('--greedy', type=bool, default=False,
                    help='Whether to run Greedy CSS subroutine.')
parser.add_argument('--save-dir', type=str, default='results',
                    help='Results folder.')
parser.add_argument('--rand-idx', type=int, default=1,
                    help='Random permuted data index.')
args = parser.parse_args()


save_dir = args.save_dir + '_{}'.format(args.rand_idx)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


# Parameters
dataset_name = args.data
dataset_folder = args.dataset_folder
rank = args.rank
sketch_size = int(rank / 3)  # for k-CSS 1,2
sparsity = int(rank / 3)  # for k-CSS 1,2

# time recorder
time_recorder_regular = {
    'gen_sketching': 0.,
    'gen_coreset': 0.,
    'k_CSS': 0.,
    'l1_reg': 0.
}

N = 1000

# Load dataset.
if dataset_name == 'synthetic':
    A = generate_synthetic_matrix(args.rank, N)
    # random shuffle column index
    _, n = A.shape
    shuffle_idx = np.random.permutation(n)
    A = A[:, shuffle_idx]
else:
    A = load_dataset(dataset_name + '_rand_{}'.format(args.rand_idx), dataset_folder)
d, n = A.shape
print('data shape: {}'.format(A.shape))

cauchy_size = int(0.5 * d)
coreset_size = 2 * rank

t1 = time.time()
S = generate_dense_cauchy_matrix(size=(cauchy_size, d))
time_recorder_regular['gen_sketching'] = time.time() - t1


def recursive_merge(C_, D_):
    if len(C_) == 1:
        return C_, D_
    (C_2, l_2), (C_1, l_1) = C_[-2], C_[-1]
    if l_2 == l_1:
        C_ = C_[:-2]
        D_2, D_1 = D_[-2], D_[-1]
        D_ = D_[:-2]
        # merge C_2, C_1 => a new strong coreset
        coreset_union = np.hstack((C_2, C_1))
        col_union = np.hstack((D_2, D_1))
        _, col_num = coreset_union.shape
        if col_num <= coreset_size:
            C_0 = coreset_union
            D_0 = col_union
        else:
            T, sel_idx = perform_l1_lewis_weight_sampling(np.transpose(col_union),
                                                          sample_rows=coreset_size,
                                                          exact=True)
            C_0 = np.matmul(coreset_union, np.transpose(T))
            D_0 = col_union[:, sel_idx]
        C_.append((C_0, l_2 + 1))
        D_.append(D_0)
        return recursive_merge(C_, D_)
    else:
        return C_, D_

t3 = time.time()
# Initialize
C = []  # list of list of coresets + level no.
D = []  # list of list of unsketched columns

# Process data stream
print('Start processing data stream...')
batch_start = 0
batch_size = 5 * rank
while batch_start < n:
    batch_end = min(batch_start + batch_size, n)
    L = A[:, batch_start:batch_end]
    M = S @ L
    C.append((M, 0))
    D.append(L)
    C, D = recursive_merge(C, D)
    batch_start = batch_end
# postprocess
C_union = np.hstack([cc for (cc, _) in C])
D_union = np.hstack(D)
_, cols_left = C_union.shape
if cols_left > coreset_size:
    T, sel_idx = perform_l1_lewis_weight_sampling(np.transpose(C_union),
                                                  sample_rows=coreset_size,
                                                  exact=True)
    coreset = np.matmul(C_union, np.transpose(T))
    sel_cols = D_union[:, sel_idx]
else:
    coreset = C_union
    sel_cols = D_union

t4 = time.time()
time_recorder_regular['gen_coreset'] = t4 - t3

# k-CSS on the single coreset left in C
print('End of data stream in {:.4f} s'.format(t4 - t3))

t5 = time.time()
if args.greedy:
    _, _, css_sel_cols = greedy_CSS_l12(coreset, args.rank)
else:
    _, css_sel_cols = regular_CSS_l12(coreset, sketch_size, sparsity,
                                      rank, only_indices=True, exact=True)

output_cols = sel_cols[:, css_sel_cols]
t6 = time.time()
time_recorder_regular['k_CSS'] = t6 - t5
print('Output cols shape: {} in {:.4f} s'.format(output_cols.shape, t6 - t5))

# Compute l1 regression error.
t_reg = time.time()
error = compute_l1_error(output_cols, A)
t_reg_end = time.time()
time_recorder_regular['l1_reg'] = t_reg_end - t_reg
print('error: {}, l1 regression in time {:.4f} s'.format(error, t_reg_end - t_reg))

# save results
kcss_method = 'greedy' if args.greedy else 'regular'
savename = os.path.join(save_dir, '{}_streaming_data_{}_rank_{}'\
            .format(kcss_method, args.data, args.rank))
with open(savename, 'wb') as f:
    pickle.dump((error, time_recorder_regular), f)

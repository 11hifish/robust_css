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
parser.add_argument('--servers', type=int, default=5,
                    help='Num. servers.')
parser.add_argument('--rank', type=int, default=100,
                    help='Rank.')
parser.add_argument('--greedy', type=bool, default=False,
                    help='Whether to run Greedy CSS subroutine.')
parser.add_argument('--save-dir', type=str, default='results',
                    help='Results folder.')
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

@ray.remote
def generate_coreset(A, S, coreset_size):
    SA = np.matmul(S, A)
    T, sel_indices = perform_l1_lewis_weight_sampling(np.transpose(SA),
                                                      sample_rows=coreset_size,
                                                      exact=False)
    SAT = np.matmul(SA, np.transpose(T))
    sel_A = A[:, sel_indices]
    return SAT, sel_A


# Parameters
dataset_name = args.data
dataset_folder = args.dataset_folder
num_servers = args.servers
rank = args.rank
sketch_size = int(args.rank / 3)  # for k-CSS 1,2
sparsity = int(args.rank / 3)  # for k-CSS 1,2
greedy = args.greedy
N = 1000

# time recorder
time_recorder = {
    'gen_sketching': 0.,
    'gen_coreset': 0.,
    'k_CSS': 0.,
    'l1_reg': 0.
}

# 1. Load dataset.
if dataset_name == 'synthetic':
    A = generate_synthetic_matrix(rank, N)
else:
    A = load_dataset(dataset_name, dataset_folder)
d, n = A.shape

cauchy_size = int(0.5 * d)
# coreset_size = max(int(0.3 * n / args.servers), 10 * args.rank)
coreset_size = 2 * args.rank

print('data shape: {}'.format(A.shape))
A_col_list = even_split_data(A, s=num_servers)
for Ai in A_col_list:
    print('Ai shape: {}'.format(Ai.shape))

t1 = time.time()
S = generate_dense_cauchy_matrix(size=(cauchy_size, d))
time_recorder['gen_sketching'] = time.time() - t1

# 2. Generate coresets.
t2 = time.time()
futures = [generate_coreset.remote(Ai, S, coreset_size) for Ai in A_col_list]

# wait for the results
all_coresets = []
all_sel_cols = []
while len(futures) > 0:
     finished, rest = ray.wait(futures)
     coreset, columns = ray.get(finished[0])
     print(coreset.shape, columns.shape)
     all_coresets.append(coreset)
     all_sel_cols.append(columns)
     futures = rest

# check that all servers are finished
# count_future_id = servers.get_count.remote()
# num = ray.get(count_future_id)

# aggregate coresets and columns
all_coresets = np.hstack(all_coresets)  # sketched dim x (coreset size x servers)
all_sel_cols = np.hstack(all_sel_cols)  # dim x (coreset size x servers)
t3 = time.time()
time_recorder['gen_coreset'] = t3 - t2
print('all_coresets shape: {}, all_sel_cols shape: {}'.format(all_coresets.shape,
                                                              all_sel_cols.shape))
print('time: {:.4f}'.format(t3 - t2))

# run k-CSS_{1, 2} on the coresets
t4 = time.time()
if args.greedy:
    _, _, css_sel_cols = greedy_CSS_l12(all_coresets, args.rank)
else:
    _, css_sel_cols = regular_CSS_l12(all_coresets, sketch_size, sparsity,
                                      rank, only_indices=True, exact=False)

# get final columns
output_cols = all_sel_cols[:, css_sel_cols]
t5 = time.time()
time_recorder['k_CSS'] = t5 - t4
print('Finished k-CSS_12 in {:.4f} s'.format(t5 - t4))
print('final columns shape: {}'.format(output_cols.shape))

# Compute l1 regression error.
# (This method computes error for the columns in parallel.)
# error = multiple_response_l1_regression_pwSGD(output_cols, A)
print('Start computing the l1 error.')
t_reg = time.time()
error = compute_l1_error(output_cols, A)
t6 = time.time()
time_recorder['l1_reg'] = t6 - t_reg
print('error: {}'.format(error))
print('finished l1 regression in {:.4f} s'.format(t6 - t_reg))

# save results
kcss_method = 'greedy' if args.greedy else 'regular'
savename = os.path.join(args.save_dir, '{}_data_{}_servers_{}_rank_{}'\
            .format(kcss_method, args.data, args.servers, args.rank))
with open(savename, 'wb') as f:
    pickle.dump((error, time_recorder), f)

import ray
ray.init(address='auto', _redis_password='5241590000000000')
import argparse
import os
import numpy as np
from code_v2.common.utils import load_dataset, even_split_data
from code_v2.common.l1_regression import *
import time
import pickle
from code_v2.common.generate_synthetic import generate_synthetic_matrix


parser = argparse.ArgumentParser(description='Uniform column selection baseline protocol.')
parser.add_argument('--data', type=str, default='techtc',
                    help='Dataset.')
parser.add_argument('--dataset-folder', type=str, default='/home/ubuntu/robust_css/dataset',
                    help='Dataset folder.')
parser.add_argument('--servers', type=int, default=5,
                    help='Num. servers.')
parser.add_argument('--rank', type=int, default=100,
                    help='Rank.')
parser.add_argument('--save-dir', type=str, default='results',
                    help='Results folder.')
args = parser.parse_args()


if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)


@ray.remote
def get_random_columns(A, num_cols):
    d, n = A.shape
    sel_idx = np.random.choice(np.arange(n), num_cols, replace=False)
    sel_A = A[:, sel_idx]
    return sel_A

def calc_num_columns_each_server(total_cols, num_servers):
    each_col_size = total_cols // num_servers
    cols_per_server = (np.ones(num_servers) * each_col_size).astype(np.int64)
    diff = total_cols - each_col_size * num_servers
    if diff > 0:
        cols_per_server[-1] += diff
    assert (np.sum(cols_per_server) == total_cols)
    return cols_per_server


# time recorder
time_recorder = {
    'get_cols': 0.,
    'l1_reg': 0.
}

N = 1000

dataset_name = args.data
# 1. Load dataset.
if dataset_name == 'synthetic':
    A = generate_synthetic_matrix(args.rank, N)
else:
    A = load_dataset(dataset_name, args.dataset_folder)
d, n = A.shape

A_col_list = even_split_data(A, s=args.servers)

t1 = time.time()
cols_per_server = calc_num_columns_each_server(args.rank, args.servers)
futures = [get_random_columns.remote(Ai, num_col)
           for (Ai, num_col) in zip(A_col_list, cols_per_server)]

# wait for the results
all_sel_cols = []
while len(futures) > 0:
     finished, rest = ray.wait(futures)
     columns = ray.get(finished[0])
     print(columns.shape)
     all_sel_cols.append(columns)
     futures = rest

all_sel_cols = np.hstack(all_sel_cols)
t2 = time.time()
print('all_sel_cols shape: {} in time {:.4f} s'.format(all_sel_cols.shape, t2 - t1))
time_recorder['get_cols'] = t2 - t1

# Compute l1 regression error.
print('Start computing the l1 error.')
t_reg = time.time()
error = compute_l1_error(all_sel_cols, A)
t_reg_end = time.time()
print('error: {}'.format(error))
print('l1 regression time: {}'.format(t_reg_end - t_reg))
time_recorder['l1_reg'] = t_reg_end - t_reg

# save results
method_name = 'uniform'
savename = os.path.join(args.save_dir, '{}_data_{}_servers_{}_rank_{}'\
            .format(method_name, args.data, args.servers, args.rank))
with open(savename, 'wb') as f:
    pickle.dump((error, time_recorder), f)

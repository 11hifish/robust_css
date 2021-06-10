import ray
ray.init(address='auto', _redis_password='5241590000000000')
from code_v2.common.utils import load_dataset, even_split_data
from code_v2.common.l1_regression import *
from code_v2.common.generate_synthetic import generate_synthetic_matrix
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
parser.add_argument('--save-dir', type=str, default='results',
                    help='Results folder.')
parser.add_argument('--rand-idx', type=int, default=1,
                    help='Random permuted data index.')
args = parser.parse_args()


dataset_name = args.data
dataset_folder = args.dataset_folder
rank = args.rank
save_dir = args.save_dir + '_{}'.format(args.rand_idx)

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

N = 1000
# Load dataset.
if dataset_name == 'synthetic':
    A = generate_synthetic_matrix(args.rank, N)
    # random shuffle columns
    _, n = A.shape
    shuffle_idx = np.random.permutation(n)
    A = A[:, shuffle_idx]
else:
    A = load_dataset(dataset_name + '_rand_{}'.format(args.rand_idx), dataset_folder)
d, n = A.shape
print('data shape: {}'.format(A.shape))

########################################
time_recorder_uniform = {
    'get_cols': 0.,
    'l1_reg': 0.
}
# Random baseline
t1 = time.time()
rand_output_cols = A[:, :rank]
for col_idx in range(rank, n):
    cur_col = A[:, col_idx]
    keep = np.random.choice([True, False])
    if keep:
        rand_replace_idx = np.random.choice(np.arange(rank))
        rand_output_cols[:, rand_replace_idx] = cur_col
t2 = time.time()
time_recorder_uniform['get_cols'] = t2 - t1
print('Random baseline get cols in {:.4f} s'.format(t2 - t1))

t_reg = time.time()
error = compute_l1_error(rand_output_cols, A)
t_reg_end = time.time()
time_recorder_uniform['l1_reg'] = t_reg_end - t_reg
print('Random error: {}, l1 regression in time {:.4f} s'.format(error, t_reg_end - t_reg))

savename = os.path.join(save_dir, 'random_streaming_data_{}_rank_{}'\
            .format(args.data, args.rank))
with open(savename, 'wb') as f:
    pickle.dump((error, time_recorder_uniform), f)

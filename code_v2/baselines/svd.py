import numpy as np
import argparse
import time
import pickle
import os
from code_v2.common.utils import load_dataset
from code_v2.common.l1_regression import *
from code_v2.common.generate_synthetic import generate_synthetic_matrix
from code_v2.baselines.rank_k_svd import rank_k_svd

parser = argparse.ArgumentParser(description='Uniform column selection baseline protocol.')
parser.add_argument('--data', type=str, default='techtc',
                    help='Dataset.')
parser.add_argument('--dataset-folder', type=str, default='/home/ubuntu/robust_css/dataset',
                    help='Dataset folder.')
parser.add_argument('--rank', type=int, default=100,
                    help='Rank.')
parser.add_argument('--save-dir', type=str, default='results',
                    help='Results folder.')
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# time recorder
time_recorder = {
    'get_approx': 0.
}


N = 1000
# 1. Load dataset.
if args.data == 'synthetic':
    A = generate_synthetic_matrix(args.rank, N)
else:
    A = load_dataset(args.data, args.dataset_folder)

# 1. Load dataset.
print('A shape: {}'.format(A.shape))

t1 = time.time()
A_approx = rank_k_svd(A, args.rank)
t2 = time.time()
time_recorder['get_approx'] = t2 - t1

error = np.sum(np.abs(A - A_approx))

# save results
method_name = 'svd'
savename = os.path.join(args.save_dir, '{}_data_{}_rank_{}'\
            .format(method_name, args.data, args.rank))
with open(savename, 'wb') as f:
    pickle.dump((error, time_recorder), f)


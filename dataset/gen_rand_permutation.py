import numpy as np
from code_v2.common.utils import load_dataset
import pickle


data = 'gene'
A = load_dataset(data, '.')
d, n = A.shape

for idx in range(1, 11):
    rand_col_idx = np.random.permutation(n)
    A_rand_idx = A[:, rand_col_idx]
    with open('{}_rand_{}.pkl'.format(data, idx), 'wb') as f:
        pickle.dump(A_rand_idx, f)


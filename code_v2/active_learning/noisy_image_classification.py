from sklearn.metrics import mean_squared_error
import os
import numpy as np
import pickle
from code_v2.baselines.simple_kcss2 import simple_kcss2
from code_v2.common.kCSS12 import regular_CSS_l12
from sklearn.linear_model import LinearRegression


rank = 300  # change # columns selected
data_path = '../../dataset'
with open(os.path.join(data_path, 'coil20_data.pkl'), 'rb') as f:
    X, labels = pickle.load(f)
# X: (1440, 1024)

num_samples, num_features = X.shape
corrupt_num = max(int(0.4 * num_features), 1)

def mask_with_noise(D):
    n_samples, n_feat = D.shape
    E = np.zeros(D.shape)
    position = np.vstack([np.random.choice(n_feat, size=corrupt_num, replace=False) for _ in range(n_samples)])
    entries = np.vstack([np.random.uniform(0, 256, size=corrupt_num) for _ in range(n_samples)])
    row_idx = np.tile(np.arange(n_samples), corrupt_num).reshape(corrupt_num, -1).T
    E[row_idx, position] = entries
    return D + E

def run_trials(trials=1):
    all_mse_random = np.zeros(trials)
    all_mse_css2 = np.zeros(trials)
    all_mse_regular = np.zeros(trials)
    for t_idx in range(trials):
        print('trial : {}'.format(t_idx))
        # split train and test
        train_size = int(num_samples * 0.8)
        train_idx = np.random.choice(num_samples, size=train_size, replace=False)
        test_idx = np.setdiff1d(np.arange(num_samples), train_idx)
        X_train = X[train_idx]
        y_train = labels[train_idx]
        X_test = X[test_idx]
        y_test = labels[test_idx]
        # perturb
        X_train = mask_with_noise(X_train)

        A = X_train.T
        _, n_cols = A.shape

        # random baseline
        sel_idx_random = np.random.choice(n_cols, size=rank, replace=False)
        # simple k-CSS2
        _, sel_idx_css2 = simple_kcss2(A, rank)
        # regular k-CSS_{1, 2}
        _, sel_idx_regular = \
                regular_CSS_l12(A, sketch_size=int(rank / 2),
                                sparsity=int(rank / 2), rank=rank, exact=True)

        def count_class_distribution(sel_y_train):
            class_ct = np.zeros(20)
            for i in range(1, 21):
                class_ct[i-1] = len(np.where(sel_y_train==i)[0])
            print(class_ct)

        # train
        model_random = LinearRegression().fit(X_train[sel_idx_random], y_train[sel_idx_random])
        model_css2 = LinearRegression().fit(X_train[sel_idx_css2], y_train[sel_idx_css2])
        model_regular = LinearRegression().fit(X_train[sel_idx_regular], y_train[sel_idx_regular])

        # perdiction
        y_random = model_random.predict(X_test)
        y_css2 = model_css2.predict(X_test)
        y_regular = model_regular.predict(X_test)

        # evaluate performance
        mse_random = mean_squared_error(y_test, y_random)
        mse_css2 = mean_squared_error(y_test, y_css2)
        mse_regular = mean_squared_error(y_test, y_regular)

        print('Random : {:.6f}'.format(mse_random))
        print('CSS_Frobenius : {:.6f}'.format(mse_css2))
        print('Regular CSS_12 : {:.6f}'.format(mse_regular))

        all_mse_random[t_idx] = mse_random
        all_mse_css2[t_idx] = mse_css2
        all_mse_regular[t_idx] = mse_regular
    print('FINAL RESULTS: ')
    print('Random: {:.6f} ({:.4f})'.format(np.mean(all_mse_random), np.std(all_mse_random)))
    print('CSS_Frobenius: {:.6f} ({:.4f})'.format(np.mean(all_mse_css2), np.std(all_mse_css2)))
    print('Regular CSS_12: {:.6f} ({:.4f})'.format(np.mean(all_mse_regular), np.std(all_mse_regular)))

run_trials(20)


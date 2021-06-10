import pickle
import os
import numpy as np


""" Data Utils. """
def load_dataset(dataname, datafolder=None):
    dataname = dataname + '.pkl'
    if datafolder is None:  # use default datafolder path
        datafolder = os.path.join(os.getcwd(), '../../dataset')
    if dataname in ['gene.pkl']:
        with open(os.path.join(datafolder, 'gene1.pkl'), 'rb') as f:
            X1 = pickle.load(f)
        with open(os.path.join(datafolder, 'gene2.pkl'), 'rb') as f:
            X2 = pickle.load(f)
        X = np.vstack((X1, X2))
        X = X[:, :5000]
        X = X[:400, :]
    else:
        with open(os.path.join(datafolder, dataname), 'rb') as f:
            X = pickle.load(f)
    return X

def even_split_data(X, s=2):
    """
        Column-wise split data matrix across s servers evenly.
        :param X: d x n data matrix
        :param s: split across s servers
        :return: A list of splitted data matrices.
	"""
    num_cols = X.shape[1]
    batch_size = num_cols // s
    Ais = []
    start_idx = 0
    while start_idx < num_cols:
        end_idx = start_idx + batch_size
        if num_cols - end_idx < batch_size:
            Ais.append(X[:, start_idx:])
            start_idx = num_cols
        else:
            Ais.append(X[:, start_idx:end_idx])
            start_idx = end_idx
    return Ais


def generate_dense_cauchy_matrix(size):
    A1 = np.random.normal(0, 1, size=size)
    A2 = np.random.normal(0, 1, size=size)
    return A1 / A2


data_to_target_rank = {
    'synthetic': np.array([10, 30, 50, 70, 90, 110]),
    'techtc': np.array([10, 30, 50, 70, 90, 110, 130, 150]),
    'gene': np.array([50, 100, 150, 200, 250, 300, 350, 400])
}

data_to_title = {
    'synthetic': 'Synthetic',
    'techtc': 'TechTC',
    'gene': 'Gene'
}


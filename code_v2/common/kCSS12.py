import numpy as np
from code_v2.common.lewis_weights import generate_OSNAP_sparse_embedding, perform_l1_lewis_weight_sampling


def regular_CSS_l12(A, sketch_size, sparsity, rank, approx_factor=25,
                    only_indices=True, exact=False):

    # 1. Sparse Embedding Matrix
    emb = generate_OSNAP_sparse_embedding(sketch_size, A.shape[0], sparsity)
    SA = np.matmul(emb, A)

    # 2. Lewis weights sampling
    S, sel_indices = perform_l1_lewis_weight_sampling(SA.T, rank,
                                                      approx_factor, exact=exact)

    if only_indices:
        return None, sel_indices
    else:
        AS_prime = np.matmul(A, S.T)
        # 3. New sparse embedding matrix for regression
        R = generate_OSNAP_sparse_embedding(sketch_size, A.shape[0], sparsity)

        # 4. Regression - not actually used for distributed protocol
        RA = np.matmul(R, A)
        RAS_prime_inv = np.linalg.pinv(np.matmul(R, AS_prime))
        return np.matmul(RAS_prime_inv, RA), sel_indices


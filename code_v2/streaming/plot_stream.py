import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from code_v2.common.utils import load_dataset, data_to_target_rank, data_to_title
from code_v2.common.generate_synthetic import generate_synthetic_matrix


data = 'synthetic'
# data = 'techtc'
# data = 'gene'
save_dir = 'str_{}_results'.format(data)
save_dir_svd = os.path.join(os.getcwd(), '../baselines/{}_svd'.format(data))
results_no = np.arange(1, 11)
target_ranks = data_to_target_rank[data]

plt_title_err = '{} L1 Error (Streaming)'.format(data_to_title[data])
plt_title_time = '{} Time (Streaming)'.format(data_to_title[data])

# load data and calc |A|_1
N = 1000
# load data and calc |A|_1
if data == 'synthetic':
    A = [generate_synthetic_matrix(rank, N) for rank in target_ranks]
    for Ai in A:
        print(Ai.shape)
    A_1norm = np.array([np.sum(np.abs(Ai)) for Ai in A])
    print('A_1norm: ', A_1norm)
else:
    A = load_dataset(data)
    print(A.shape)
    A_1norm = np.sum(np.abs(A))


def extract_time_recorder(time_rec, method_name):
    if method_name in ['greedy', 'regular']:
        if time_rec is None:
            return 0
        return time_rec['gen_sketching'] + time_rec['gen_coreset'] + time_rec['k_CSS']
    elif method_name in ['uniform', 'random', 'uniform_single']:
        return time_rec['get_cols']
    elif method_name == 'svd':
        return time_rec['get_approx']
    else:
        return None

def load_target_streaming_results(results_no, target_ranks, method_name, save_dir):
    all_errors = []
    all_times = []
    for rno in results_no:
        folder = 'results_{}'.format(rno)
        rno_errors = np.zeros(len(target_ranks))
        rno_times = np.zeros(len(target_ranks))
        for rank_idx in range(len(target_ranks)):
            rank = target_ranks[rank_idx]
            savefile = '{}_streaming_data_{}_rank_{}'\
                    .format(method_name, data, rank)
            savename = os.path.join(save_dir, folder, savefile)
            with open(savename, 'rb') as f:
                error, time_rec = pickle.load(f)
            time_takes = extract_time_recorder(time_rec, method_name)
            rno_errors[rank_idx] = error
            rno_times[rank_idx] = time_takes
        all_errors.append(rno_errors)
        all_times.append(rno_times)
    return np.vstack(all_errors), np.vstack(all_times)

def load_svd_baseline_results(target_ranks, save_dir):
    all_errors = np.zeros(len(target_ranks))
    all_times = np.zeros(len(target_ranks))
    for rank_idx in range(len(target_ranks)):
        savefile = 'svd_data_{}_rank_{}'.format(data, target_ranks[rank_idx])
        savename = os.path.join(save_dir, savefile)
        with open(savename, 'rb') as f:
            error, time_rec = pickle.load(f)
        all_errors[rank_idx] = error
        all_times[rank_idx] = extract_time_recorder(time_rec, 'svd')
    return all_errors, all_times

regular_errs, regular_times = load_target_streaming_results(results_no, target_ranks,
                                                  'regular', save_dir)

greedy_errs, greedy_times = load_target_streaming_results(results_no, target_ranks,
                                                  'greedy', save_dir)

random_errs, random_times = load_target_streaming_results(results_no, target_ranks,
                                                  'random', save_dir)
svd_errs, svd_times = load_svd_baseline_results(target_ranks, save_dir_svd)

# post process
regular_errs_mean = np.mean(regular_errs / A_1norm, axis=0)
greedy_errs_mean = np.mean(greedy_errs / A_1norm, axis=0)
random_errs_mean = np.mean(random_errs / A_1norm, axis=0)
svd_errs = svd_errs / A_1norm

regular_errs_std = np.std(regular_errs / A_1norm, axis=0)
greedy_errs_std = np.std(greedy_errs / A_1norm, axis=0)
random_errs_std = np.std(random_errs / A_1norm, axis=0)

regular_times_mean = np.mean(regular_times, axis=0)
greedy_times_mean = np.mean(greedy_times, axis=0)
random_times_mean = np.mean(random_times, axis=0)

regular_times_std = np.std(regular_times, axis=0)
greedy_times_std = np.std(greedy_times, axis=0)
random_times_std = np.std(random_times, axis=0)

## Plot error
mk_size = 20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# baselines
ax1.scatter(target_ranks, random_errs_mean, marker='^', s=mk_size)
ax1.errorbar(target_ranks, random_errs_mean,
            yerr=random_errs_std,
            label='uniform', alpha=0.5)
ax1.scatter(target_ranks, svd_errs, marker='D', s=mk_size)
ax1.errorbar(target_ranks, svd_errs,
            label='svd', alpha=0.5)
# k-CSS
ax1.scatter(target_ranks, regular_errs_mean, s=mk_size)
ax1.errorbar(target_ranks, regular_errs_mean,
            yerr=regular_errs_std,
            label='regular', alpha=0.5)
ax1.scatter(target_ranks, greedy_errs_mean, s=mk_size)
ax1.errorbar(target_ranks, greedy_errs_mean,
            yerr=greedy_errs_std,
            label='greedy', alpha=0.5)
ax1.set_xticks(target_ranks)
ax1.grid(alpha=0.5)
# ax.set_ylim([0, 1])
ax1.legend()
ax1.set_title(plt_title_err)
ax1.set_xlabel('# Columns Selected')
ax1.set_ylabel('Error Ratio')
# plt.show()


## Plot time
# Baselines
ax2.scatter(target_ranks, random_times_mean, marker='^', s=mk_size)
ax2.errorbar(target_ranks, random_times_mean,
            yerr=random_times_std,
            label='uniform', alpha=0.5)
ax2.scatter(target_ranks, svd_times, marker='D', s=mk_size)
ax2.errorbar(target_ranks, svd_times,
            label='svd', alpha=0.5)
# k-CSS
ax2.scatter(target_ranks, regular_times_mean, s=mk_size)
ax2.errorbar(target_ranks, regular_times_mean,
            yerr=regular_times_std,
            label='regular', alpha=0.5)
ax2.scatter(target_ranks, greedy_times_mean, s=mk_size)
ax2.errorbar(target_ranks, greedy_times_mean,
            yerr=greedy_times_std,
            label='greedy', alpha=0.5)
ax2.set_xticks(target_ranks)
ax2.grid(alpha=0.5)
ax2.legend()
ax2.set_title(plt_title_time)
ax2.set_xlabel('# Columns Selected')
ax2.set_ylabel('Time (seconds)')
# plt.show()
plt.savefig('{}_streaming'.format(data), bbox_inches='tight',
    pad_inches=0.02)


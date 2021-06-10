import os
from code_v2.common.utils import data_to_target_rank

data = 'synthetic'  # change dataset name
random_idx = 1  # change here for multiple runs
result_folder_data = 'str_{}_results'.format(data)
if not os.path.isdir(result_folder_data):
    os.mkdir(result_folder_data)
result_folder = 'results_{}'.format(random_idx)
save_dir = os.path.join(result_folder_data, 'results')
ranks = data_to_target_rank[data]
print('data : {}, ranks: {}'.format(data, ranks))

all_cmds = ''
for k in ranks:
     all_cmds += 'python3 random_stream.py --data {} --save-dir {} --rand-idx {} --rank {}; sleep 3; '\
         .format(data, save_dir, random_idx, k)

print(all_cmds)
os.system(all_cmds)

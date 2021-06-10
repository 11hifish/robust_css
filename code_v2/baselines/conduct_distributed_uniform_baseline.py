import os
from code_v2.common.utils import data_to_target_rank

data = 'synthetic'  # change dataset name
exp_no = 1  # change here for multiple runs
result_folder_data = '{}_uniform'.format(data)
if not os.path.isdir(result_folder_data):
    os.mkdir(result_folder_data)
result_folder = 'results_{}'.format(exp_no)
res_folder = os.path.join(result_folder_data, result_folder)
if not os.path.isdir(res_folder):
    os.mkdir(res_folder)
ranks = data_to_target_rank[data]
print('data : {}, ranks: {}'.format(data, ranks))

all_cmds = ''
for k in ranks:
     all_cmds += 'python3 uniform_distributed.py --data {} --save-dir {} --rank {}; sleep 3; '\
         .format(data, res_folder, k)

print(all_cmds)
os.system(all_cmds)

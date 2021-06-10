import os
from code_v2.common.utils import data_to_target_rank

data = 'synthetic'  # change dataset name
result_folder_data = '{}_svd'.format(data)
if not os.path.isdir(result_folder_data):
    os.mkdir(result_folder_data)
ranks = data_to_target_rank[data]
print('data : {}, ranks: {}'.format(data, ranks))

all_cmds = ''
for k in ranks:
     all_cmds += 'python3 svd.py --data {} --save-dir {} --rank {}; sleep 3; '\
         .format(data, result_folder_data, k)

print(all_cmds)
os.system(all_cmds)

import os
import glob
import torch
import random
import numpy as np

def get_next_run_index(save_folder):
    existing_runs = [d for d in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, d)) and d.startswith('run')]
    run_indices = [int(d.replace('run', '').split('_')[0]) for d in existing_runs if d.replace('run', '').split('_')[0].isdigit()]
    if run_indices:
        return max(run_indices) + 1
    else:
        return 0

def aggregate_results(save_folder, output_file='average_results.txt', logger=None):
    target_oa_list = []
    source_oa_list = []
    target_class_acc_list = []
    source_class_acc_list = []
    
    result_files = glob.glob(os.path.join(save_folder, 'run*_seed*', 'result_tr180_val0.txt'))
    print(f"Found {len(result_files)} result files in {save_folder}")
    
    if not result_files:
        if logger:
            logger.warning(f"在{save_folder}中未找到任何result_tr180_val0.txt文件")
        return 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
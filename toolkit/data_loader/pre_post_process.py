import numpy as np
import cv2
import sys
sys.path.append('..')
from toolkit.base_function import io_disp_read
from toolkit.backup.dataset_function import generate_file_lists

def generate_some_args(dataset_name):
    max_disp = None
    max_disp_sr = None
    if 'VirtualRoad' in dataset_name:
        max_disp = 200
        if 'pt' in dataset_name:
            max_disp_sr = 60
        else:
            max_disp_sr = 60
    if 'idd' in dataset_name:
        max_disp = None
    if 'realroad' in dataset_name:
        max_disp = 200
        if 'pt' in dataset_name:
            max_disp_sr = 80
        else:
            max_disp_sr = 250
    return max_disp,max_disp_sr

def generate_appended_file_dirs(dataset,len_):
    disp_list_noc = np.zeros((len_))
    disp_list_fd = np.zeros((len_))
    if dataset in ['realroad','realroad_pt']:
        disp_list_noc = generate_file_lists(dataset = dataset,if_train=True,method='BGNet')['disp_list']
    if dataset in ['realroad','realroad_pt']:
        disp_list_noc = generate_file_lists(dataset = dataset,if_train=True,method='noc_mask')['disp_list']
    if 'VirtualRoad' in dataset:
        disp_list_fd = generate_file_lists(dataset = dataset,if_train=True,method='pt_disp')['disp_list']
        disp_list_noc = generate_file_lists(dataset = dataset,if_train=True,method='gt')['disp_list']
    return disp_list_noc,disp_list_fd


# post process the disp result, such as mask the noc area, and add the pt fd disp to recover
def results_post_process(result,dataset,dataset_type,data,noc_dir=None,pt_fd_dir=None):
    if 'middlebury' in dataset:
        if 'im0' in data['left_dir'][0]:
            noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
        elif 'view1' in data['left_dir'][0]:
            noc_mask = cv2.imread(data['left_dir'][0].replace('view1.png','noc_mask.png'),-1)
        noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
        result = result*noc_mask                
    elif 'MiddEval3' in dataset and dataset_type == 'train':
        noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
        noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
        result = result*noc_mask  
    elif 'realroad' in dataset:
        if "pt" in dataset:
            result = result+io_disp_read(data['disp_dir'][0])
        # print('mask the disp with {}'.format(noc_dir))
        mask_disp = io_disp_read(noc_dir) 
        result[mask_disp==0]=0
    elif 'VirtualRoad' in dataset:       
        assert noc_dir.split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0] == pt_fd_dir.split('/')[-1].split('.')[0]
        if 'pt' in dataset:
            pt = io_disp_read(pt_fd_dir)
            # avoid adding pt_fd disp for sparse matching
            pt[result==0]=0
            result = result+pt
        # print('mask the disp with {}'.format(noc_dir))
        # if result.shape[0] == ori_H and result.shape[1] == ori_W:
        gt_disp = io_disp_read(noc_dir)
        result[gt_disp==0]=0
    return result

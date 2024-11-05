import torch
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import sys
sys.path.append('..')
from toolkit.base_function import single_image_warp,io_disp_read
from toolkit.backup.dataset_function import generate_file_lists,virtual_road_clear,realroad_clear
from toolkit.backup.evaluator import evaluator_disp,evaluator_image
import json

# evaluate disp
def evaluate_supervised(pre_list,gt_list,evaluator_class=evaluator_disp,save_dir='image_evaluation/delete.json',method='delete',resize_gt=False,left_list=None,additional_mask_list=None,title=None):
    evaluator = evaluator_class()
    
    for index in range(len(pre_list)):
        gt_disp = io_disp_read(gt_list[index])
        print('inputing gt image {}, shape {}'.format(gt_list[index],gt_disp.shape))
        pre_disp = io_disp_read(pre_list[index])
        # pre_disp[np.isnan(pre_disp)] = 0
        print('inputing pre image {}, shape {}'.format(pre_list[index],pre_disp.shape))

        H0,W0 = gt_disp.shape
        H,W = pre_disp.shape

        if not (gt_disp.shape[0] == pre_disp.shape[0] and gt_disp.shape[1] == pre_disp.shape[1]):
            if resize_gt:
                gt_disp = cv2.resize(gt_disp*W/W0, (W,H), interpolation=cv2.INTER_NEAREST)
            else:
                pre_disp = cv2.resize(pre_disp*W0/W, (W0,H0), interpolation=cv2.INTER_NEAREST)
        assert gt_disp.shape == pre_disp.shape, 'numbers of gt files {} and predicted files {} are not equal'.format(gt_disp.shape, pre_disp.shape)

        # input to evaluator
        gt_disp = torch.from_numpy(gt_disp)
        pre_disp = torch.from_numpy(pre_disp)
        if not additional_mask_list is None:
            # disp_noc = additional_mask_list[index].replace('road_seg_mask','disparity_noc/disp').replace('png','npy')
            mask = cv2.imread(additional_mask_list[index],-1)
            # disp_noc = io_disp_read(disp_noc)
            # mask[mask>0]=1
            # gt_disp = gt_disp*mask
            # pre_disp = pre_disp*mask
            gt_disp[mask==0] = 0
            pre_disp[mask==0] = 0
            # gt_disp[disp_noc==0] = 0
            # pre_disp[disp_noc==0] = 0
        evaluator.input_gt_mask(gt_disp)
        evaluator.input_data(pre_disp)
        
        if not left_list is None:
            image = cv2.imread(left_list[index],cv2.IMREAD_GRAYSCALE).flatten()
            if resize_gt:
                image = cv2.resize(image, (W,H), interpolation=cv2.INTER_NEAREST)
            evaluator.input_ori_image(image)

        # break

    # print("start evaluate")
    value_dic_img,value_dic_mean = evaluator.process()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,method+'_mean.json'), "a") as f:
        f.write('\r')
        if not title is None:
            f.write(title)
            f.write('\r')
        f.write(json.dumps(str(value_dic_mean)))
    with open(os.path.join(save_dir,method+'_each.json'), "a") as f:
        f.write('\r')
        f.write(json.dumps(str(value_dic_img)))
    print(value_dic_mean)
    print('save in {}'.format(os.path.join(save_dir,method+'_mean.json')))
    return value_dic_mean

def evaluate_unsupervised(file_lst_l,file_lst_r,file_lst_d,file_lst_d_gt,save_dir,method,additional_mask_list=None,title=None):        
    evaluator = evaluator_image()
    ssim_list = []

    for i in range(len(file_lst_r)):
        print('warping right image {} and disparity image {}'.format(file_lst_r[i],file_lst_d[i]))
        disp = io_disp_read(file_lst_d[i])
        if (np.mean(disp)<100) and not file_lst_d_gt is None: # pt disp
            fd = io_disp_read(file_lst_d_gt[i])
            disp = disp + fd
        disp = torch.from_numpy(disp)
        right = cv2.imread(file_lst_r[i])
        left = cv2.imread(file_lst_l[i])

        warped_right = single_image_warp(right,disp,'right')
        warped_right[disp==0,:]=0
        left[disp==0,:]=0
        # warped_right[disp==0,:]=left[disp==0,:]

        if not additional_mask_list is None:
            mask = cv2.imread(additional_mask_list[i])
            mask[mask>0]=1
            left = left*mask
            warped_right = warped_right*mask

        warp_save_dir = 'generate_images/'+"/".join(file_lst_l[i].split('/')[2:-2])
        name = file_lst_l[i].split('/')[-1]
        if not os.path.exists(warp_save_dir):
            os.makedirs(warp_save_dir)
        cv2.imwrite(os.path.join(warp_save_dir,name),warped_right)

        evaluator.input_image(left)
        evaluator.input_data(warped_right)
        ssim_value = ssim(left.astype(np.uint8),warped_right.astype(np.uint8),channel_axis=2)
        ssim_list.append(ssim_value)
        # break

    # print("start evaluate")
    value_dic_img,value_dic_mean = evaluator.process()
    value_dic_img['ssim'] = ssim_list
    value_dic_mean['ssim'] = np.mean(ssim_list)

    # print(value_dic_img)
    # print(value_dic_mean)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,method+'_mean.json'), "a") as f:
        f.write('\r')
        if not title is None:
            f.write(title)
            f.write('\r')
        f.write(json.dumps(value_dic_mean))
    with open(os.path.join(save_dir,method+'_each.json'), "a") as f:
        f.write('\r')
        f.write(json.dumps(value_dic_img))
    print('save in {}'.format(os.path.join(save_dir,method+'_mean.json')))

def realroad_evaluate(method,dataset='realroad'):
    file_path_dic = generate_file_lists(dataset = dataset,if_train=True,method=method)
    save_dir = "image_evaluation/{}".format(dataset)
    evaluate_unsupervised(file_path_dic['left_list'],file_path_dic['right_list'],file_path_dic['disp_list'],None,save_dir,method) # evaluator_disp, 

def VirtualRoad_evaluate(method,dataset='VirtualRoad',noc=True):
    file_path_dic = generate_file_lists(dataset = dataset,if_train=True,method='gt')
    gt_list = file_path_dic['disp_list']
    left_list = file_path_dic['left_list']
    right_list = file_path_dic['right_list']
    file_path_dic = generate_file_lists(dataset = dataset,if_train=True,method=method)
    pre_list = file_path_dic['disp_list']
    assert len(gt_list) == len(left_list) == len(right_list) == len(pre_list), str(len(gt_list), len(left_list), len(right_list), len(pre_list))  
    save_dir = "image_evaluation/{}".format(dataset)
    road_mask_list = []
    for i in range(len(left_list)):
        dir = left_list[i].replace('rgb_front_left','road_seg_mask').replace('pure_road_left','road_seg_mask')
        road_mask_list.append(dir)
    value_dic_mean_ = evaluate_supervised(pre_list,gt_list,evaluator_disp,save_dir,method,additional_mask_list = road_mask_list,title='road mask') 
    return value_dic_mean_


if __name__ == '__main__':    
    
    for network in ['graft','BGNet','Unimatch','LacGwc']:
        realroad_evaluate(network+'_D3Stereo')
        # VirtualRoad_evaluate(network+'_D3Stereo')
        
    virtual_road_clear()   
    realroad_clear()   
        
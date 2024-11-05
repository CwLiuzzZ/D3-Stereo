import numpy as np
from pylab import *
import torch
import torch.nn.functional as F
from scipy import interpolate

import sys
sys.path.append('..')
sys.path.append('../..')
from toolkit.base_function import disp_vis,SparseDisp2Points,Points2SparseDisp

#!/usr/bin/env python
# coding: utf-8

# input Conf: cos_sim 1~1: higher~better
def seed_growth(Conf,sparse_disp=None,points=None,name='delete',subpixel=True,Conf_r=None,left_image=None,filling=True,display=False,D3_rad_=1,D3_SR_ = [-1,0,1]):
    global D3_rad 
    global D3_sr

    D3_rad = D3_rad_ # 1 2 3 
    D3_sr = D3_SR_ # [-1,0,1] [--2,1,0,1,1] [-3,-2,-1,0,1,2,3] 

    # print('D3_rad',D3_rad)
    # print('D3_sr',D3_sr)

    Conf = 2-2*Conf # cos_sim -> L2 sim: 0:4: lower~better

    H = Conf.shape[0]
    W = Conf.shape[1]
    
    max_disp = Conf.shape[-1]
    assert not (sparse_disp is None and points is None)
    # sparse_disp = sparse_disp.float()
    if sparse_disp is None:
        sparse_disp = Points2SparseDisp(H,W,points[0],points[1])

    left_disp = seed_growth_algorithm(H,W,max_disp,Conf,sparse_disp.clone(),name=name,subpixel=subpixel,display=display)
    if display:
        disp_vis('generate_images/seed/left.png',left_disp) # ,left_disp.shape[-1]*0.3

    if Conf_r is None:
        return left_disp

    Conf_r = 2-2*Conf_r
    if points is None:
        points = SparseDisp2Points(sparse_disp)
    sparse_disp = Points2SparseDisp(H,W,points[1],points[0])
    sparse_disp = torch.flip(-sparse_disp,[-1])

    right_disp = seed_growth_algorithm(H,W,max_disp,Conf_r,sparse_disp,name=name,subpixel=subpixel)
    if display:
        disp_vis('generate_images/seed/right.png',right_disp) # ,right_disp.shape[-1]*0.2
    dispmap_lrc = LRDCC(left_disp,right_disp,mode = 'right',T=1)    # input tensor, output numpy [H,W]


    if display:
        disp_vis('generate_images/seed/lrc.png',dispmap_lrc) # ,dispmap_lrc.shape[-1]*0.3
    if not filling:
        return dispmap_lrc
    
    zero_mask = torch.zeros(dispmap_lrc.shape,device='cuda')
    zero_mask[dispmap_lrc==0]=1
    dispmap_filled = disparity_filling(H,W,dispmap_lrc,zero_mask) 

    if display:
        disp_vis('generate_images/seed/fill.png',dispmap_filled) # ,dispmap_filled.shape[-1]*0.3
    disp = get_FBS(H,W,left_image,dispmap_filled,zero_mask,rad=1,gamma_d=8,gamma_r=10,cycle=5)
    # disp_vis('generate_images/seed/filt.png',disp,disp.shape[-1]/10)

    grid = torch.arange(0, disp.shape[1], device='cuda').float().unsqueeze(0).expand(disp.shape[0],disp.shape[1]) # [H,W]: H * 0~W
    disp = torch.where(disp>grid,grid,disp)

    return disp
    

def seed_growth_algorithm(image_y,image_x,max_disp,Conf,disp,name='delete',subpixel=True,display=False):

    # print('---seed growth for {}---'.format(name))
    Conf[:,:,:2]=4
    disp[disp>max_disp-1]=0
    disp = disp.float()
    # pixel without disparity is 1, otherwise 0
    points_with_disp = torch.zeros(disp.shape,device='cuda')
    points_with_disp[disp>0]=1
    
    nn_Unfold=torch.nn.Unfold(kernel_size=((2*D3_rad+1),(2*D3_rad+1)),dilation=1,padding=D3_rad,stride=1)
    
    start_ratio_th = 0.5 #
    thr_list = []
    while start_ratio_th<2:
        thr_list.append(start_ratio_th)
        start_ratio_th = start_ratio_th*1.1
        
    count=0
    count_quit=0
    new_list = []
    while True:
        count = count+1
        search_disparities = make_search_pixels_adversarial(disp,nn_Unfold) # [H,W], [n,27]
        search_disparities = torch.clip(search_disparities,1,max_disp-2) # [H,W,8]

        search_conf = torch.gather(Conf,-1,search_disparities)
        search_conf_l  = torch.gather(Conf,-1,search_disparities+1)
        search_conf_r  = torch.gather(Conf,-1,search_disparities-1)
        # exclude the non local extreme (LE) disparities 
        # find the most confidence LE disparity
        search_conf[search_conf>=torch.min(search_conf_l,search_conf_r)]=5 # turn to LE_conf (local extreme)
        search_disparities[search_conf==5]=0
        conf, conf_index = torch.min(search_conf, dim=-1) # score and number [n]
        update_disp = torch.gather(search_disparities,-1,conf_index.unsqueeze(-1)).squeeze().float() #[n], from index to disparity candidate with maximum confidence
        update_pixels = int(torch.sum(torch.abs(update_disp-disp)>0))

        if len(new_list)<30:
            new_list.append(update_pixels)
        else:
            del new_list[0]
            new_list.append(update_pixels)

        if display:
            disp_vis('generate_images/seed2/refine_iter_{}.png'.format(str(count)),disp.detach().cpu().numpy())
            # disp_vis('generate_images/seed2/refine_iter_{}.png'.format(str(count)),disp.detach().cpu().numpy(),int(image_x*0.3))
            # print('disparity adversarial refinement, iteration: {}, update pixels: {}'.format(count,update_pixels))

        disp = update_disp

        if count_quit>170:
            break
        if update_pixels<=3:
            break
        if update_pixels > np.mean(new_list):
            count_quit+=5
        if update_pixels<150:
            count_quit+=1
        elif update_pixels<50:
            count_quit+=2

    if subpixel:
        disp = subpixel_enhancement(disp,Conf,max_disp)  # tensor

    return disp

def make_search_pixels(disp,points_with_disp,nn_Unfold):
    search_pixels = torch.zeros(disp.shape,device='cuda')
    search_disparities = nn_Unfold(disp.unsqueeze(0).unsqueeze(0)) # [1,9,H*W]
    search_disparities = search_disparities.view((2*D3_rad+1)**2,disp.shape[-2],disp.shape[-1]).permute(1,2,0) #[H,W,9]
    search_disparities_ = torch.max(search_disparities,dim=-1)[0]

    search_pixels[search_disparities_>0] = 1
    search_pixels = search_pixels*(1-points_with_disp)
    search_pixels = search_pixels.bool()
    search_disparities = search_disparities[search_pixels,:] # [n,9] # ignore the pixels with disparity
    return search_pixels,search_disparities.long()

def make_search_pixels_adversarial(disp,nn_Unfold):
    image_y,image_x = disp.shape
    search_disparities = nn_Unfold(disp.unsqueeze(0).unsqueeze(0)) # [1,9,H*W]
    search_disparities = search_disparities.view((2*D3_rad+1)**2,image_y,image_x).permute(1,2,0) #[H,W,9]
    search_disparities = torch.cat([search_disparities + sr for sr in D3_sr],dim=-1) # [H,W,24]
    return search_disparities.long()

def isolate_check(disp,nn_Unfold,pool,thr=1):
    disp_new = disp.clone()
    image_y,image_x = disp.shape
    exist=torch.zeros(disp.shape,device='cuda')
    # growth form image margin
    exist[0,:]=1
    exist[-1,:]=1
    exist[:,0]=1
    exist[:,-1]=1

    count=0
    while True:
        count+=1
        check = pool(exist.unsqueeze(0)).squeeze() # [H,W]
        check[check>0]=1
        check = check-exist

        disp_new=disp*exist
        disp_new[disp_new==0]=-1-thr
        disp_unfold = nn_Unfold(disp_new.float().unsqueeze(0).unsqueeze(0)) # [1,9,H*W]
        disp_unfold = disp_unfold.view((2*D3_rad+1)**2,image_y,image_x).permute(1,2,0) #[H,W,9]

        diff = disp*check-torch.max(disp_unfold) # [H,W]
        check[diff>=thr]=0
        exist += check
        if torch.sum(check)==0:
            break
        # if count%1==0:
        #     cv2.imwrite('generate_images/seed2/isolate_check_{}.png'.format(count),exist.detach().cpu().numpy()*255)
        #     disp_vis('generate_images/seed2/isolate_check.png',disp.detach().cpu().numpy(),130)
    print('isolate check, pixels left: {}'.format(image_x*image_y-torch.sum(exist)))
    return exist

def subpixel_enhancement(dispmap,conf,max_disp):
    dispmap = dispmap.long()
    # dispmap = torch.clip()
    mask_valid_pixel = torch.ones(size=dispmap.shape)
    mask_valid_pixel[dispmap<=1]=0
    mask_valid_pixel[dispmap>=max_disp-2]=0

    conf0 = conf.gather(2, dispmap.unsqueeze(2)).squeeze()
    dispmap_ = dispmap - 1
    dispmap_[dispmap_<0]=0
    conf1 = conf.gather(2, dispmap_.unsqueeze(2)).squeeze()
    dispmap_ = dispmap + 1
    dispmap_[dispmap_>max_disp-1]=max_disp-1
    conf2 = conf.gather(2, dispmap_.unsqueeze(2)).squeeze()

    # if 4 exists in conf0,1,2
    conf_stack = torch.stack([conf0,conf1,conf2],dim=0)
    _ = torch.max(conf_stack,dim=0)[0]
    mask_valid_pixel[_==4]=0
    # Local Extreme
    _ = conf0 - torch.min(conf_stack,dim=0)[0]
    mask_valid_pixel[_>0]=0
    mask_valid_pixel = mask_valid_pixel.bool()
    
    dispmap = dispmap.float()
    # sub pixel enforcement
    update = dispmap + (conf1-conf2)/(2*conf1-4*conf0+2*conf2)

    # invalid pixel
    dispmap[mask_valid_pixel] = update[mask_valid_pixel]
    # print('sub_pixel_disp nan: ',torch.isnan(dispmap).any())

    if torch.isnan(dispmap).any():
        dispmap = torch.where(torch.isnan(dispmap), torch.full_like(dispmap, 0), dispmap)
        # print('nan exist in dispmap')
        # exit()

    return dispmap

def disparity_filling(H,W,disp,zero_mask):
    disp_candidate = torch.ones((H,W,4)).cuda()*W
    disp_scan_c1_bak = (torch.ones((H))*W).cuda()
    disp_scan_c2_bak = disp_scan_c1_bak.clone()

    _ = disp.clone()

    for i in range(W):
        disp_scan1 = disp[:,i] # [H]
        disp_update_1 = torch.where(disp_scan1==0,disp_scan_c1_bak,disp_scan1)
        disp_candidate[:,i:,0] = disp_update_1.unsqueeze(-1) # [H,?]
        disp_scan_c1_bak = torch.where(disp_scan1>0,disp_scan1,disp_scan_c1_bak)

        j = W-i
        disp_scan2 = disp[:,j-1] # [H]
        disp_update_2 = torch.where(disp_scan2==0.,disp_scan_c2_bak,disp_scan2)
        disp_candidate[:,:j,1] = disp_update_2.unsqueeze(-1)
        disp_scan_c2_bak = torch.where(disp_scan2>0,disp_scan2,disp_scan_c2_bak)

    disp_update,_ = torch.min(disp_candidate,-1)
    disp = disp+disp_update*zero_mask

    return disp
    
def LRDCC(disp_l,disp_r,mode,T):
    disp_r_flip = torch.flip(disp_r,[1]);
    disp_r = warp_process(disp_r_flip,disp_l,mode) 
    disp_l[torch.abs(disp_l-disp_r)>T]=0
    return disp_l

# mode represents the type of input img
def warp_process(img, disp,mode,torch_=False):
    assert mode in ['left','right']
    if mode == 'left':
        # should be negative disparity
        if torch.mean(disp)<0:
            disp=-disp
    elif mode == 'right':
        # should be positive disparity
        if torch.mean(disp)>0:
            disp=-disp
    assert img.shape[-1] == disp.shape[-1]
    assert img.shape[-2] == disp.shape[-2]
    disp = (disp/disp.shape[1]).float()
    disp = disp.unsqueeze(0).unsqueeze(0)
    
    if len(img.shape)==2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape)==3:
        img = img.unsqueeze(0)
        
    batch_size, _, height, width = img.shape

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    # print('shifts',torch.mean(x_shifts))
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)
    if output.shape[1]==1:
        return output[0][0]
    else:
        return output[0]
    
def get_FBS(img_y,img_x,img,disp,mask,rad=1,gamma_d=1,gamma_r=1,cycle=1):
    disp_raw = disp*(-(mask-1))
    img = F.interpolate(img, size=(img_y,img_x), mode='bilinear', align_corners=True)
    img_pad = F.pad(img, [rad,rad,rad,rad],mode='replicate') # [1,1, H，W]
    unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=1,padding=0,stride=1)
    img_unfold = unfold(img_pad)
    img_unfold = img_unfold.view((2*rad+1)**2,img_y,img_x).permute(1,2,0) #[H,W,9]
    w_r = torch.exp(-torch.square(img_unfold - img.squeeze().unsqueeze(-1))/(gamma_r**2)) # [H,W,9]
    
    w = w_r # [H,W,9]

    for i in range(cycle):
        disp = disp.unsqueeze(0).unsqueeze(0) # [1,1,H,W]
        disp = F.pad(disp,[rad,rad,rad,rad],mode='replicate') # [1,1,H+1,W+1]
        
        disp = unfold(disp)
        disp = disp.view((2*rad+1)**2,img_y,img_x).permute(1,2,0) #[H,W,9]
        disp = disp*w

        disp = torch.sum(disp,dim=-1) # [H,W]
        disp /= torch.sum(w,dim=-1) # [H,W]
        disp = disp_raw+disp*mask

    return disp

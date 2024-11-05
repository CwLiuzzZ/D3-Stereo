#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:46:43 2021

@author: kutalmisince
"""
import cv2 as cv2
import torch
import torch.nn.functional as F
from toolkit.base_function import sim_remap,SparseDisp2Points,sim_construct,sim_restore,get_pt_disp,single_image_warp,get_ncc_sim
from toolkit.backup.seed_growth import seed_growth
from DFM.DeepFeatureMatcher_NCC import refine_points_2D_2_ncc
import os
import time
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


#############
## with PT ##
#############
def match_inference(args,img_A, img_B,img_A_dir,img_B_dir,feature_extractor,layer_size,bidirectional=True,padding=16,display=False):
        
        ratio_th=args.ratio_th
        RBF_cycle=args.RBF_cycle
        pt=args.pt
        rad = args.BF_rad
        
        '''
        layer_size: down_size of the feature layer compared with the original resolution
        H: homography matrix warps image B onto image A, compatible with cv2.warpPerspective,
        and match points on warped image = H * match points on image B
        H_init: stage 0 homography
        '''
        _,_,H,W = img_A.shape
        draw_A = cv2.imread(img_A_dir) # [H,W,C]
        draw_B = cv2.imread(img_B_dir) # [H,W,C]
        draw_A=cv2.resize(draw_A,(W,H),interpolation=cv2.INTER_LINEAR)     
        draw_B=cv2.resize(draw_B,(W,H),interpolation=cv2.INTER_LINEAR)     
        _,ori_W,_ = draw_A.shape
        down_size = int(ori_W/300)

        # transform into pytroch tensor and pad image to a multiple of 16
        inp_A, padding_A = transform(img_A,padding) # inp_A: image after padding, padding_A: padding parameters
        inp_B, padding_B = transform(img_B,padding) 

        filt_image = torch.from_numpy(cv2.imread(img_A_dir,cv2.CV_8UC1)).cuda().float().unsqueeze(0).unsqueeze(0)
        filt_image = F.interpolate(filt_image, size=(H,W), mode='bilinear', align_corners=True)
        filt_image = F.pad(filt_image, [0,padding_A[0],0,padding_A[1]],mode='constant', value=0.0)
        assert filt_image.shape[-1] == inp_A.shape[-1] and filt_image.shape[-2] == inp_A.shape[-2] ,str(filt_image.shape) + '_' +str(inp_A.shape)

        filt_image_r = torch.from_numpy(cv2.imread(img_B_dir,cv2.CV_8UC1)).cuda().float().unsqueeze(0).unsqueeze(0)
        filt_image_r = F.interpolate(filt_image_r, size=(H,W), mode='bilinear', align_corners=True)
        filt_image_r = F.pad(filt_image_r, [0,padding_A[0],0,padding_A[1]],mode='constant', value=0.0)
        filt_image_r = torch.flip(filt_image_r,[-1])

        # get acitvations # get learned representations
        activations_A = feature_extractor(inp_A)
        activations_B = feature_extractor(inp_B)


        assert len(ratio_th)==len(layer_size),str(ratio_th)+' '+str(layer_size)
        assert len(activations_A) == len(activations_B) == len(RBF_cycle), str(len(activations_A))+' '+str(len(activations_B))+' '+str(len(RBF_cycle))

        if pt:
            sim, sim_r = sim_construct(activations_A[-1].squeeze(),activations_B[-1].squeeze(),LR=True) # H,W,W 
            sim = get_FBS(filt_image,sim,rad=rad,gamma_r=args.BF_i,cycle = 3) 
            sim_r = get_FBS(filt_image_r,sim_r,rad=rad,gamma_r=args.BF_i,cycle = 3)
            points_A, points_B = dense_feature_matching(sim, ratio_th[-1], bidirectional)
            
            seed_disp = seed_growth(sim,points=(points_A,points_B),name='pt',Conf_r=sim_r,left_image=filt_image,display=False, D3_rad_ = args.D3_rad,D3_SR_ = args.D3_SR,filling=True) 
            points_A, points_B = SparseDisp2Points(seed_disp.clone(),remove_margin=True) #[W,H]
            points_A = points_A.detach().cpu().numpy()
            points_B = points_B.detach().cpu().numpy()
            occ_match = np.where(points_A[0,:] < sim.shape[1]/6)[0]
            points_A = np.delete(points_A,occ_match,1)
            points_B = np.delete(points_B,occ_match,1)
            pt_disp = get_pt_disp(sim.shape[0],sim.shape[1],points=[points_A[0],points_A[1],(points_A[0]-points_B[0])],offset=args.pt_offset/layer_size[-1])
            pt_disp = cv2.resize(pt_disp*layer_size[-1],(W,H),interpolation=cv2.INTER_LINEAR)   
            pt_disp = (torch.tensor(pt_disp)).cuda()
            img_B = single_image_warp(img_B,pt_disp,tensor_output=True)
            # img_B_ = img_B.detach().cpu().squeeze(0).permute(1,2,0).numpy()
            # cv2.imwrite('generate_images/DFM_2_warped_right.png',img_B_*255) # vis for warped right image 
            inp_B, padding_B = transform(img_B,padding) 
            activations_B = feature_extractor(inp_B)


        # normalize and reshape feature maps
        sim, sim_r = sim_construct(activations_A[-1].squeeze(),activations_B[-1].squeeze(),LR=True,down_size=down_size) # H,W,W 
        sim = get_FBS(filt_image,sim,rad=rad,gamma_r=args.BF_i,cycle = RBF_cycle[-1]) 
        sim_r = get_FBS(filt_image_r,sim_r,rad=rad,gamma_r=args.BF_i,cycle = RBF_cycle[-1])
        points_A, points_B = dense_feature_matching(sim, ratio_th[-1], bidirectional)
        seed_disp = seed_growth(sim,points=(points_A,points_B),name='match',subpixel=False,Conf_r=sim_r,left_image=filt_image,D3_rad_ = args.D3_rad,D3_SR_ = args.D3_SR,display=False) 

        refine_index=0
        # points should be long tensor
        for k in range(len(layer_size)-2,-1,-1):
            refine_index+=1
            points_A, points_B = refine_points_2D(points_A, points_B, activations_A[k].squeeze(), activations_B[k].squeeze(),layer_size[k+1]/layer_size[k])
            sim,sim_r = sim_construct(activations_A[k].squeeze(), activations_B[k].squeeze(),LR=True,down_size=down_size) # H,W,W disp
            sim = get_FBS(filt_image,sim,rad=rad,gamma_r=args.BF_i,cycle = RBF_cycle[k])
            sim_r = get_FBS(filt_image_r,sim_r,rad=rad,gamma_r=args.BF_i,cycle = RBF_cycle[k]) 

            seed_disp = seed_growth(sim,points=(points_A,points_B),name='refine_{}'.format(refine_index),Conf_r=sim_r,left_image=filt_image,display=True, D3_rad_ = args.D3_rad,D3_SR_ = args.D3_SR) 

            if not k==0:
                points_A, points_B = SparseDisp2Points(seed_disp.clone(),remove_margin=True)

        if args.NCC_enhance:
            inp_A = inp_A*255
            inp_B = inp_B*255
            points_A, points_B = refine_points_2D_2_ncc(points_A, points_B, inp_A,inp_B,layer_size[0],rad)
            sim = get_ncc_sim(inp_A, inp_B, ncc_rad = 3,max_disp=int(W/down_size))
            sim_r = get_ncc_sim(inp_B.flip([-1]), inp_A.flip([-1]), ncc_rad = 3,max_disp=int(W/down_size))
            sim = get_FBS(filt_image,sim,rad=1,cycle = 5,gamma_r=args.BF_i) 
            sim_r = get_FBS(filt_image_r,sim_r,rad=1,cycle = 5,gamma_r=args.BF_i)             
            seed_disp = seed_growth(sim,points=(points_A,points_B),name='refine_NCC',Conf_r=sim_r,left_image=filt_image,display=False,D3_rad_ = args.D3_rad,D3_SR_ = args.D3_SR)
            points_A, points_B = SparseDisp2Points(seed_disp.clone(),remove_margin=True)
            points_A, points_B = rescale_points((points_A,points_B),(padding_A,padding_B),1,inp_A.shape)
            disp = seed_disp
        else:
            points_A, points_B = rescale_points((points_A,points_B),(padding_A,padding_B),layer_size[0],inp_A.shape)
            disp = F.interpolate((seed_disp*layer_size[0]).unsqueeze(0).unsqueeze(0), size=(seed_disp.shape[0]*layer_size[0],seed_disp.shape[1]*layer_size[0]), mode='bilinear', align_corners=True).squeeze() # [H,W]
        disp = disp[:H,:W]

        if pt:
            disp = disp + pt_disp
            points_B[0,:] = points_B[0,:] - pt_disp[points_A[1,:],points_A[0,:]]
        return {'points_A':points_A.detach().cpu().numpy(),'points_B':points_B.detach().cpu().numpy(),'disp':disp}

def dense_feature_matching(sim, ratio_th,bidirectional=True):
    sim_ = sim_restore(sim)
    sim_ = sim_remap(sim_) 
    points_A,points_B=mnn_ratio_matcher(sim_, ratio_th, bidirectional) # [H,W]
    return points_A, points_B
  
def mnn_ratio_matcher(sim, ratio=0.8, bidirectional = True):
    # print('ratio',ratio)
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    device = sim.device

    nns_sim, nns = torch.topk(sim, 2, dim=2,largest=True) # score and number [H,W,2]
    nns_dist = 2 - 2 * nns_sim # cosine similarity --> (L2 similarity)**2; (-1,1) --> (4,0) smaller -- similar
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:,:,0] / (nns_dist[:,:,1] + 1e-8) # N[H,W]
    # Save first NN and match similarity.
    nn12 = nns[:,:,0] # [H,W]
    # match_sim = nns_sim[:,:,0] # [H,W]

    nns_sim, nns = torch.topk(sim.transpose(2,1), 2, dim=2,largest=True) # score and number [H,W,2]
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:,:,0] / (nns_dist[:,:,1] + 1e-8) # [H,W]
    # Save first NN.
    nn21 = nns[:,:,0] # [H,W]

    # if not bidirectional, do not use ratios from 2 to 1 # LRC
    ratios21[:,:] *= 1 if bidirectional else 0
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[1], device=device).unsqueeze(0).expand(sim.shape[0],sim.shape[1]).contiguous() # [H,W]: H * 0~W
    lrc = torch.diagonal(nn21[:,nn12]).t() # [H,W]

    mask = torch.min(ids1 == lrc, torch.min(ratios12 <= ratio, torch.diagonal(ratios21[:,nn12]).t() <= ratio)) # [H,W] discard ratios21 to get the same results with matlab

    # discard correspondences at image margin
    mask[:,0]=False
    mask[:,-1]=False
    mask[0,:]=False
    mask[-1,:]=False
    mask[:,1]=False
    mask[:,-2]=False
    mask[1,:]=False
    mask[-2,:]=False
    mask[nn12<=1]=False
    mask[nn12>=(mask.shape[1]-2)]=False
    mask[ids1<=nn12]=False

    # disp = (ids1-nn12)*mask
    # disp[disp<0]=0
    # # print(torch.sum())
    # match_sim = match_sim[disp>0] 
    # # ratios12 = ratios12[disp>0]      
    # (points_A_,points_B_) = SparseDisp2Points(disp) # [2,n]

    _ = mask.nonzero() # [n,2] [H,W]
    points_A = torch.zeros(_.shape).cuda()
    points_B = torch.zeros(_.shape).cuda()
    points_A[:,0] = _[:,1]
    points_B[:,0] = nn12[mask]
    points_A[:,1] = _[:,0]
    points_B[:,1] = _[:,0]
    points_A = points_A.t().long()
    points_B = points_B.t().long()

    # points_B_2 = points_B.clone()
    # points_B_2[0,:] = nns[:,:,1][disp>0]
    # match_sim_2 = nns_sim[:,:,1][disp>0]  
    #  
    return points_A,points_B
    # return match_sim.data.cpu(),points_A,points_B

def transform(img,padding_n=16):
    
    '''
    Convert given uint8 numpy array to tensor, perform normalization and 
    pad right/bottom to make image canvas a multiple of padding_n

    Parameters
    ----------
    img : nnumpy array (uint8)

    Returns
    -------
    img_T : torch.tensor
    (pad_right, pad_bottom) : int tuple 

    '''
    
    # transform to tensor and normalize
    # T = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Lambda(lambda x: x.to("cuda:0")),
    #                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                 std=[0.229, 0.224, 0.225])
    #                         ])
    
    # zero padding to make image canvas a multiple of padding_n
    pad_right = padding_n - img.shape[-1] % padding_n if img.shape[-1] % padding_n else 0
    pad_bottom = padding_n - img.shape[-2] % padding_n if img.shape[-2] % padding_n else 0
    
    padding = torch.nn.ZeroPad2d([0, pad_right, 0, pad_bottom])
    
    # convert image
    #img_T = padding(T(img.astype(np.float32))).unsqueeze(0)
    img_T = padding(img)
    # img_T = padding(T(img)).unsqueeze(0)
    return img_T, (pad_right, pad_bottom)  


def get_FBS(img,cost_volume,rad=1,gamma_d=1,gamma_r=7,cycle=1):

    # print('rad',rad)
    # print('cycle',cycle)

    if cycle > 0:
        # gamma_d = 15
        gamma_r = torch.tensor(gamma_r) # BGNet:7; Unimatch:7; LacGwc:7; AANet:4; PSMNet:7; Cre:9

        assert rad in [1,2,3,4,5,6]
        img_y,img_x,D =cost_volume.shape # H,W,D
        pad = rad

        cost_volume = cost_volume.permute(2,0,1).unsqueeze(0) # [1,D,H,W]
        mask = cost_volume==-1

        img = F.interpolate(img, size=(img_y,img_x), mode='bilinear', align_corners=True)
        img_pad = F.pad(img, [pad,pad,pad,pad],mode='replicate') # [1,1, H，W]
        unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=1,padding=0,stride=1)
        img_unfold = unfold(img_pad)
        img_unfold = img_unfold.view((2*rad+1)**2,img_y,img_x).permute(1,2,0) #[H,W,9]
        # gamma_r = torch.std(img_unfold,dim=-1,keepdim=True)*2
        w_r = torch.exp(-torch.square(img_unfold - img.squeeze().unsqueeze(-1))/(2*torch.square(gamma_r))) # [H,W,9]
        # if rad == 1:
        #     w_d = torch.tensor([[2,1,2],[1,0,1],[2,1,2]]).cuda()
        # elif rad == 2:
        #     w_d = torch.tensor([[8,5,4,5,8],[5,2,1,2,5],[4,1,0,1,4],[5,2,1,2,5],[8,5,4,5,8]]).cuda()
        # w_d = torch.exp(-w_d/gamma_d**2).view(-1).unsqueeze(0).unsqueeze(0) # [1,1,9]
        # w = (w_r*w_d).unsqueeze(-2) # [H,W,1,9]
        w = w_r.permute(-1,0,1).unsqueeze(0) # [1,9,H,W]
        w /= torch.sum(w,dim=1).unsqueeze(0) # [1,9,H,W]

        # torch.cuda.empty_cache()
        for i in range(cycle):
            cost_volume = F.pad(cost_volume,[pad,pad,pad,pad],mode='replicate') # [1,D,H+rad,W+rad]
            cost_volume = unfold(cost_volume)
            cost_volume = cost_volume.view(D,(2*rad+1)**2,img_y,img_x) #[D,9,H,W]
            # print(cost_volume.shape)
            for i in range(D): # disp dim
                cost_volume[i,:,:,:] = cost_volume[i,:,:,:]*w[0]
            # cost_volume = cost_volume*w
            cost_volume = torch.sum(cost_volume,dim=1).unsqueeze(0) # [1,D,H,W]
            cost_volume[mask]=-1
        
        cost_volume = cost_volume.squeeze(0).permute(1,2,0)

    return cost_volume

def rescale_points(points,padding_points,scale,inp_A_shape):
    points_A = points[0]
    points_B = points[1]
    padding_A = padding_points[0]
    padding_B = padding_points[1]

    # points_A = (points_A + 0.5) * scale - 0.5
    # points_B = (points_B + 0.5) * scale - 0.5
    points_A = points_A * scale 
    points_B = points_B * scale

    # points_A = points_A.double()
    
    # optional: because padding at the right and the bottom, so can just remove them
    in_image = torch.logical_and(points_A[0, :] < (inp_A_shape[-1] - padding_A[0]), points_A[1, :] < (inp_A_shape[-2] - padding_A[1]))
    in_image = torch.logical_and(in_image, 
                                    torch.logical_and(points_B[0, :] < (inp_A_shape[-1] - padding_B[0]), points_B[1, :] < (inp_A_shape[-2] - padding_B[1])))
    points_A = points_A[:, in_image]
    points_B = points_B[:, in_image]

    return points_A,points_B

def draw_points(draw_img,points,save_path,gt_disp,ratio=1):
    img_A = draw_img.copy()

    error_threshold = 1.5
    error_threshold = error_threshold*ratio

    # img_B = draw_img[1].copy()
    points_A = points[0].long()
    points_B = points[1].long()

    gt = gt_disp[points_A[1,:],points_A[0,:]]
    error = points_A[0,:]-points_B[0,:]-gt
    # error_mask = torch.logical_and(torch.abs(error)>1,torch.abs(error)>gt*0.05)
    error_mask = torch.abs(error)>error_threshold
    error_mask_1 = torch.logical_and(error_mask,error>0)
    error_mask_2 = torch.logical_and(error_mask,error<0)
    points_A_1 = points_A[:,~error_mask]
    points_A_2 = points_A[:,error_mask_1]
    points_A_3 = points_A[:,error_mask_2]

    for l in range(points_A_1.shape[1]):  
        x = int(points_A_1[0,l])
        y = int(points_A_1[1,l])
        cv2.circle(img_A, (x,y), 3, (255,0,0),1) # blue for correct correspondences
    for l in range(points_A_2.shape[1]):  
        x = int(points_A_2[0,l])
        y = int(points_A_2[1,l])
        cv2.circle(img_A, (x,y), 3, (0,0,255),1) # red for wrong correspondences with big rpedicted disparity
    for l in range(points_A_3.shape[1]):  
        x = int(points_A_3[0,l])
        y = int(points_A_3[1,l])
        cv2.circle(img_A, (x,y), 3, (0,255,255),1) # yellow for wrong correspondences with small rpedicted disparity
    cv2.imwrite(save_path, img_A)
   

def refine_points_2D(points_A: torch.Tensor, points_B: torch.Tensor, activations_A: torch.Tensor, activations_B: torch.Tensor,scale):

    # normalize and reshape feature maps
    d1 = activations_A / activations_A.square().sum(0).sqrt().unsqueeze(0) # [C,H,W]
    d2 = activations_B / activations_B.square().sum(0).sqrt().unsqueeze(0) # [C,H,W]

    # get number of points
    ch = d1.size(0)
    num_input_points = points_A.size(1) # [2,n] n=92748
    assert not num_input_points == 0
    
    # actual points_A[1,:] = points_B[1,:]
    # upsample points
    points_A = (points_A*scale).long()
    points_B = (points_B*scale).long()

    # neighborhood to search [width,height]
    neighbors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]],device='cuda')
    # neighbors_n = torch.tensor([[-1, 0], [-1, 1], [2, 0], [2, 1],[-2, 0], [-2, 1], [3, 0], [3, 1]])
    neighbors_n = torch.tensor([[-1, 0], [-1, 1], [2, 0], [2, 1]])
    # allocate space for scores
    scores = torch.zeros(num_input_points, neighbors.size(0), neighbors.size(0)) # [n,4,4]
    scores_1n = torch.zeros(num_input_points, neighbors.size(0), neighbors_n.size(0)) # [n,4,4]
    scores_2n = torch.zeros(num_input_points, neighbors.size(0), neighbors_n.size(0)) # [n,4,4]
    # for each point search the refined matches in given [finer] resolution

    for i, n_A in enumerate(neighbors):
        for m, nn_B in enumerate(neighbors_n):
            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].view(ch, -1) # [128.n]
            act_Bn = d2[:, points_B[1, :] + nn_B[1], points_B[0, :] + nn_B[0]].view(ch, -1) # [128.n]
            # compute mse
            ans = act_A * act_Bn # [128.n]
            scores_1n[:, i, m] = torch.sum(ans, 0)

            # get features in the given neighborhood
            act_A = d2[:, points_B[1, :] + n_A[1], points_B[0, :] + n_A[0]].view(ch, -1) # [128.n]
            act_Bn = d1[:, points_A[1, :] + nn_B[1], points_A[0, :] + nn_B[0]].view(ch, -1) # [128.n]

            # compute mse
            ans = act_A * act_Bn # [128.n]
            scores_2n[:, i, m] = torch.sum(ans, 0)

        for j, n_B in enumerate(neighbors):
            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].view(ch, -1) # [128.n]
            act_B = d2[:, points_A[1, :] + n_B[1], points_B[0, :] + n_B[0]].view(ch, -1) # [128.n]
            # compute mse
            ans = act_A * act_B # [128.n]
            scores[:, i, j] = torch.sum(ans, 0)

    # strict
    scores_1n = torch.max(torch.max(scores_1n,dim=-1)[0],-1)[0] # [n]
    scores_2n = torch.max(torch.max(scores_2n,dim=-1)[0],-1)[0] # [n] 

    # select the best match
    score_A2B, match_A2B = torch.max(scores, dim=2) #[n,4]
    
    scores = scores.transpose(2,1)
    # select the best match
    score_B2A, match_B2A = torch.max(scores, dim=2)
    
    # check for unique matches and apply ratio test
    ind_A = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_A2B).flatten() # n*4
    ind_B = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_B2A).flatten() # n*4
    ind = torch.arange(num_input_points * neighbors.size(0)) # n*4
    
    # torch.min
    mask = (ind_B[ind_A] == ind).view(num_input_points, -1) # [n,4]
    # local extrema
    mask = torch.logical_and(mask,score_A2B > scores_1n.unsqueeze(-1))
    mask = torch.logical_and(mask,score_B2A > scores_2n.unsqueeze(-1))
    # consistency
    mask = torch.logical_and(mask,(torch.mean(score_A2B,dim=-1) > scores_1n).unsqueeze(-1))
    mask = torch.logical_and(mask,(torch.mean(score_B2A,dim=-1) > scores_2n).unsqueeze(-1))
    # mask = torch.logical_and(mask,torch.max(ratio_A2B, ratio_B2A) < ratio_th)

    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    score_A2B[~mask] = -1
    # each input point can generate max two output points, so discard the two with highest SSE 
    _, discard = torch.topk(score_A2B, 2, dim=1,largest=False)
    for i in range(2):
        mask[torch.arange(num_input_points), discard[:, i]] = 0
    
    # x & y coordiates of candidate match points of A
    neighbors = neighbors.cuda()
    x = points_A[0, :].repeat(4, 1).t() + neighbors[:, 0].repeat(num_input_points, 1)
    y = points_A[1, :].repeat(4, 1).t() + neighbors[: ,1].repeat(num_input_points, 1)
    
    refined_points_A = torch.stack((x[mask], y[mask]))
    
    # x & y coordiates of candidate match points of A
    x = points_B[0, :].repeat(4, 1).t() + neighbors[:, 0][match_A2B]
    y = points_B[1, :].repeat(4, 1).t() + neighbors[:, 1][match_A2B]
    
    refined_points_B = torch.stack((x[mask], y[mask]))

    mask = torch.logical_and(refined_points_A[1,:] == refined_points_B[1,:],refined_points_A[0,:] > refined_points_B[0,:],)
    refined_points_A = refined_points_A[:,mask]
    refined_points_B = refined_points_B[:,mask]

    # if the number of refined matches is not enough to estimate homography,
    # but number of initial matches is enough, use initial points
    if refined_points_A.shape[1] < 4 and num_input_points > refined_points_A.shape[1]:
        refined_points_A = points_A
        refined_points_B = points_B

    return refined_points_A, refined_points_B


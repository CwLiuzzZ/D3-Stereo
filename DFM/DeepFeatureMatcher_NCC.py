#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:46:43 2021

@author: kutalmisince
"""
import torch
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

  
#     ######################
#     # original 2D refine #
#     ######################
def refine_points_2D_2_ncc(points_A: torch.Tensor, points_B: torch.Tensor, left: torch.Tensor, right: torch.Tensor,scale,ncc_rad):

    # normalize and reshape feature maps
    _,ch,H,W = left.shape
    n = (ncc_rad*2+1)**2

    # ncc initial 
    ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

    left_avg = ncc_pool(left_padded).squeeze() # [C,H,W]
    right_avg = ncc_pool(right_padded).squeeze() # [C,H,W]
    left_unfold = ncc_Unfold(left_padded).view(ch,n,H,W) # [ch,n,H,W]
    right_unfold = ncc_Unfold(right_padded).view(ch,n,H,W) # [ch,n,square,H,W]

    # get number of points
    num_input_points = points_A.size(1) # [2,n] n=92748
    assert not num_input_points == 0
    # actual points_A[1,:] = points_B[1,:]
    # upsample points
    points_A = (points_A*scale).long()
    points_B = (points_B*scale).long()

    # neighborhood to search [width,height]
    neighbors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    neighbors_n = torch.tensor([[-1, 0], [-1, 1], [2, 0], [2, 1]])
    # allocate space for scores
    scores = torch.zeros(num_input_points, neighbors.size(0), neighbors.size(0)) # [n,4,4]
    scores_1n = torch.zeros(num_input_points, neighbors.size(0), neighbors_n.size(0)) # [n,4,4]
    scores_2n = torch.zeros(num_input_points, neighbors.size(0), neighbors_n.size(0)) # [n,4,4]
    # for each point search the refined matches in given [finer] resolution
    for i, n_A in enumerate(neighbors):
        for m, nn_B in enumerate(neighbors_n):
            # get features in the given neighborhood
            minus_A = left_unfold[:, :, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]] - left_avg[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].unsqueeze(1) # [ch,n,N]
            minus_B = right_unfold[:, :, points_B[1, :] + nn_B[1], points_B[0, :] + nn_B[0]] - right_avg[:, points_B[1, :] + nn_B[1], points_B[0, :] + nn_B[0]].unsqueeze(1) # [ch,n,N]
            _1 = torch.sum(minus_A*minus_B,1) # [ch,N]
            _2 = torch.sqrt(torch.sum(torch.square(minus_A),1)*torch.sum(torch.square(minus_B),1)) # [ch,N]
            scores_1n[:, i, m] = torch.mean(_1/_2, 0)

            # get features in the given neighborhood
            minus_A = right_unfold[:, :, points_A[1, :] + nn_B[1], points_A[0, :] + nn_B[0]] - right_avg[:, points_A[1, :] + nn_B[1], points_A[0, :] + nn_B[0]].unsqueeze(1) # [ch,n,N]
            minus_B = left_unfold[:, :, points_B[1, :] + n_A[1], points_B[0, :] + n_A[0]] - left_avg[:, points_B[1, :] + n_A[1], points_B[0, :] + n_A[0]].unsqueeze(1) # [ch,n,N]
            _1 = torch.sum(minus_A*minus_B,1) # [ch,N]
            _2 = torch.sqrt(torch.sum(torch.square(minus_A),1)*torch.sum(torch.square(minus_B),1)) # [ch,N]
            scores_2n[:, i, m] = torch.mean(_1/_2, 0)

        for j, n_B in enumerate(neighbors):

            # get features in the given neighborhood
            minus_A = left_unfold[:, :, points_A[1, :] + n_A[1], points_A[0, :]]- left_avg[:, points_A[1, :] + n_A[1], points_A[0, :]].unsqueeze(1) # [ch,n,N]
            minus_B = right_unfold[:, :, points_A[1, :] + n_B[1], points_B[0, :] + n_B[0]] - right_avg[:, points_A[1, :] + n_B[1], points_B[0, :] + n_B[0]].unsqueeze(1) # [ch,n,N]
            _1 = torch.sum(minus_A*minus_B,1) # [ch,N]
            _2 = torch.sqrt(torch.sum(torch.square(minus_A),1)*torch.sum(torch.square(minus_B),1)) # [ch,N]
            scores[:, i, j] = torch.mean(_1/_2, 0)


    scores_1n = torch.max(torch.max(scores_1n,dim=-1)[0],-1)[0] # [n]
    scores_2n = torch.max(torch.max(scores_2n,dim=-1)[0],-1)[0] # [n]

    # select the best match
    score_A2B, match_A2B = torch.max(scores, dim=2) #[n,4,2]
    
    scores = scores.transpose(2,1)
    # select the best match
    score_B2A, match_B2A = torch.max(scores, dim=2)
    
    # check for unique matches and apply ratio test
    ind_A = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_A2B).flatten() # n*4
    ind_B = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_B2A).flatten() # n*4
    ind = torch.arange(num_input_points * neighbors.size(0)) # n*4
    
    # torch.min
    mask = (ind_B[ind_A] == ind).view(num_input_points, -1) # [n,4]
    # local
    mask = torch.logical_and(mask,score_A2B > scores_1n.unsqueeze(-1))
    mask = torch.logical_and(mask,score_B2A > scores_2n.unsqueeze(-1))
    # global
    mask = torch.logical_and(mask,(torch.mean(score_A2B,dim=-1) > scores_1n).unsqueeze(-1))
    mask = torch.logical_and(mask,(torch.mean(score_B2A,dim=-1) > scores_2n).unsqueeze(-1))

    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    # score_A2B[~mask] = -1
    
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

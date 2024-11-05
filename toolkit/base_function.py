import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import re
import time
from pylab import math
import sys

def dic_merge(dic1,dic2):
    if not len(list(dic1.keys()))==0:
        assert dic1.keys() == dic2.keys(),dic1.keys()+'---'+dic2.keys()
        ans = {}
        for key in dic2.keys():
            ans[key] = dic1[key]+dic2[key]
    else:
        return dic2
    return ans

def single_image_warp(img,disp,mode='right', tensor_output = False):
    '''
    function:
        warp single image with disparity map to another perspective
    input:
        img: image; should be 2D or 3D array
        disp: disparity map; should be 2D array or tensor
        mode: perspective of the input image
    output:
    '''
    assert mode in ['left','right']

    if not isinstance(img, torch.Tensor):
        if len(img.shape)==3 and (img.shape[-1]==3 or img.shape[-1]==1): # img in [H,W,channel]
            img = np.transpose(img,(2,0,1))
        img = img.astype(np.float32)
        img = torch.from_numpy(img) # [C,H,W] or [H,W]

    if not isinstance(disp, torch.Tensor):
        disp = torch.tensor(disp)
    disp = (disp/disp.shape[1]).float()

    if mode == 'left':
        # should be negative disparity
        if torch.mean(disp)<0:
            disp=-disp
    elif mode == 'right':
        # should be positive disparity
        if torch.mean(disp)>0:
            disp=-disp
    
    # disp = torch.from_numpy(disp/disp.shape[1]).float()
    
    assert img.shape[-1] == disp.shape[-1], img.shape+' and '+disp.shape
    assert img.shape[-2] == disp.shape[-2], img.shape+' and '+disp.shape

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
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)

    if tensor_output: 
        return output # [1,C,H,W]

    if output.shape[1]==1:
        output = output[0][0].detach().numpy()
    else:
        output = output[0].detach().numpy()
    if len(output.shape)==3:
        output = np.transpose(output,(1,2,0))

    return output

# COLORMAP_JET, COLORMAP_PARULA, COLORMAP_MAGMA, COLORMAP_PLASMA, COLORMAP_VIRIDIS
# disp should be [H,W] numpy.array
def disp_vis(save_dir,disp,max_disp=None,min_disp=0,colormap=cv2.COLORMAP_JET,inverse=False):
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if max_disp is None:
        disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        # disp=-disp+255
    else:
        disp = np.clip(disp,min_disp,max_disp)
        disp = 255 * (disp-min_disp)/(max_disp-min_disp)
    disp=disp.astype(np.uint8)
    if inverse:
        disp = 255 - disp
    disp = cv2.applyColorMap(disp,colormap)
    if not save_dir is None:
        cv2.imwrite(save_dir,disp)
    else:
        return disp

def dirs_walk(dir_list):
    '''
    output:
        all the files in dir_list
    '''
    file_list = []
    for dir in dir_list:
        paths = os.walk(dir)
        for path, dir_lst, file_lst in paths:
            file_lst.sort()
            for file_name in file_lst:
                file_path = os.path.join(path, file_name)
                file_list.append(file_path)
    file_list.sort()
    return file_list 

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims,divide,mode=None):
        self.mode = mode
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
        pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
        # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        # if self.mode is None:
        #     return [F.pad(x, self._pad) for x in inputs]
        # else:
        #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def io_disp_read(dir):   
    '''
    function: load disparity map from disparity file
    input:
        dir: dir of disparity file
    ''' 
    # load disp from npy
    if dir.endswith('npy'):
        disp = np.load(dir)
    # load disp from middlebury2014 or ETH3D
    # elif ('middlebury' in dir and dir.endswith('pfm')) or ('ETH3D' in dir and dir.endswith('pfm'))or ('Sceneflow' in dir and dir.endswith('pfm')):
    elif dir.endswith('pfm'):
        with open(dir, 'rb') as pfm_file:
            header = pfm_file.readline().decode().rstrip()
            channels = 3 if header == 'PF' else 1

            dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")

            scale = float(pfm_file.readline().decode().rstrip())
            if scale < 0:
                endian = '<' # littel endian
                scale = -scale
            else:
                endian = '>' # big endian

            disp = np.fromfile(pfm_file, endian + 'f')
            disp = np.reshape(disp, newshape=(height, width, channels))  

            disp[np.isinf(disp)] = 0

            disp = np.flipud(disp)    
            # disp = np.flipud(disp).astype('uint8') 

            if channels == 1:
                disp = disp.squeeze(2)
    elif 'middlebury' in dir and dir.endswith('png'):
        disp = cv2.imread(dir, -1)
    # load disp from KITTI
    elif 'KITTI' in dir:
        disp = cv2.imread(dir, cv2.IMREAD_ANYDEPTH) / 256.0
    elif 'real_road' in dir and 'disp' in dir:
        _bgr = cv2.imread(dir)
        R_ = _bgr[:, :, 2]
        G_ = _bgr[:, :, 1]
        B_ = _bgr[:, :, 0]
        normalized_= (R_ + G_ * 256. + B_ * 256. * 256.) / (256. * 256. * 256. - 1)
        disp = 500*normalized_
    else:
        raise ValueError('unknown disp file type: {}'.format(dir))
    disp = disp.astype(np.float32)
    return disp

# recover the resolution of "input" from "reference"
def reso_recover(input,reference_dir):
    ori_img = cv2.imread(reference_dir)
    img_H,img_W = ori_img.shape[0],ori_img.shape[1]
    input = cv2.resize(input*img_W/input.shape[1],(img_W,img_H),interpolation=cv2.INTER_LINEAR)
    return input

def seed_record(img_W,img_H,Key_points_coordinate,disp_min=0):
    if disp_min < 0:
        disp_min = 0
    image1_seed = np.zeros((img_H,img_W))
    image2_seed = np.zeros((img_H,img_W))
    for i in Key_points_coordinate:
        l_x = int(round(i[0],0))
        l_y = int(round(i[1],0))
        r_x = int(round(i[2],0))
        r_y = int(round(i[3],0))
        if l_x-r_x > disp_min and l_y==r_y:
            image1_seed[l_y][l_x]=l_x-r_x
            image2_seed[r_y][r_x]=l_x-r_x
    return image1_seed,image2_seed

# resample the sim: coordinate based <--> disparity based
def sim_remap(sim):
    image_x = sim.shape[-1]
    w_base,d_base = torch.meshgrid(torch.from_numpy(np.arange(image_x)),torch.from_numpy(np.arange(image_x)))    
    d_base = w_base - d_base
    w_base = ((w_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    d_base = ((d_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    coords = torch.stack((d_base, w_base), dim=3)
    # re-coordinate sim from [H,W,W] to [H,W,D]
    sim = F.grid_sample(sim.unsqueeze(0), 2*coords - 1, mode='nearest',
                            padding_mode='zeros',align_corners=True).squeeze()
    sim[sim==0] = -1
    return sim

# return points in [n,2] [width,height]
def SparseDisp2Points(disp,remove_margin=False):

    if remove_margin:
        # remove the correspondences at the image margin
        disp[:,0]=0
        disp[:,-1]=0
        disp[0,:]=0
        disp[-1,:]=0
        grid = torch.arange(0, disp.shape[1], device='cuda').unsqueeze(0).expand(disp.shape[0],disp.shape[1]) # [H,W]: H * 0~W
        disp[disp==grid]=0
        # disp[grid<=disp+1]=0

        # disp[:,1]=0
        # disp[:,-2]=0
        # disp[1,:]=0
        # disp[-2,:]=0
        # grid = torch.arange(0, disp.shape[1], device='cuda').unsqueeze(0).expand(disp.shape[0],disp.shape[1]) # [H,W]: H * 0~W
        # disp[disp==(grid-1)]=0

    _ = disp.nonzero() # [n,2]

    disp = disp[disp>0]
    points_A = torch.zeros(_.shape).cuda()
    points_B = torch.zeros(_.shape).cuda()
    points_A[:,1] = _[:,0]
    points_B[:,1] = _[:,0]
    points_A[:,0] = _[:,1]
    points_B[:,0] = points_A[:,0]-disp

    return (points_A.long().t(),points_B.long().t())

# return disp
def Points2SparseDisp(H,W,points_A,points_B):

    # point [W,H]
    disp=torch.zeros(size=(H,W)).long().cuda()
    disp[points_A[1,:],points_A[0,:]]=points_A[0,:]-points_B[0,:]
    return disp


# confidence matrix initialization 
def sim_construct(feature_A,feature_B,remap=True,LR = False,R=False,down_size=None):
    if not R:
        d1 = feature_A/torch.sqrt(torch.sum(torch.square(feature_A), 0)).unsqueeze(0) # [C,H,W]
        d2 = feature_B/torch.sqrt(torch.sum(torch.square(feature_B), 0)).unsqueeze(0) # [C,H,W]
        # d1 = d1.detach().cpu()
        # d2 = d2.detach().cpu()
        sim = torch.einsum('ijk,ijh->jkh', d1, d2) # [H,W,W] 166,240,240
        if LR:
            sim_l = sim_remap(sim)

            sim_r = sim.permute(0,2,1)
            # sim_r = torch.flip(sim_r,[-1,-2])
            sim_r = sim_r.flip([-1,-2])
            sim_r = sim_remap(sim_r)
            # # print(sim_r[100,W-1-90,10])

            # d1 = torch.flip(d1,[-1])
            # d2 = torch.flip(d2,[-1])
            # sim_r = torch.einsum('ijk,ijh->jkh', d2, d1) # [H,W,W] 166,240,240
            # sim_r = sim_remap(sim_r)
            # print(sim_r[100,W-1-90,10])

            if not down_size is None:
                sim_l = sim_down_size(sim_l,down_size)
                sim_r = sim_down_size(sim_r,down_size)
            return sim_l.cuda(),sim_r.cuda()
        if remap:
            # return similarity volume with the 3rd dim at the disp
            sim = sim_remap(sim)
            if not down_size is None:
                sim = sim_down_size(sim,down_size)
            return sim

        else:
            # return similarity volume with the 3rd dim at the coordinate
            return sim 
    elif R:
        d1 = feature_A/torch.sqrt(torch.sum(torch.square(feature_A), 0)).unsqueeze(0) # [C,H,W]
        d2 = feature_B/torch.sqrt(torch.sum(torch.square(feature_B), 0)).unsqueeze(0) # [C,H,W]            
        d1 = torch.flip(d1,[-1])
        d2 = torch.flip(d2,[-1])
        sim_r = torch.einsum('ijk,ijh->jkh', d2, d1) # [H,W,W] 166,240,240
        sim_r = sim_remap(sim_r)
        if not down_size is None:
            sim_r = sim_down_size(sim_r,down_size)
        return sim_r

def get_pt_disp(image_y,image_x,points=None,disp=None,offset=None):
    # disp should be numpy
    assert not isinstance(disp, torch.Tensor)
    assert points is None or disp is None
    if points is None:
        _ = disp.nonzero()
        u = _[1]
        v = _[0]
        dxs = disp[_]
    else:
        u = points[0] # column # width
        v = points[1] # row # height
        dxs = points[2] # disp
    PT_disp = getPT(u,v,dxs,image_y,image_x)
    for j in range(image_y):
        ans = np.min(PT_disp[j,:])
        PT_disp[j,:] = ans
    if not offset is None:
        PT_disp = PT_disp - offset
    PT_disp[PT_disp<0]=0
    return PT_disp

def getPT(u,v,d,vmax,umax):
    v_map_1 = np.mat(np.arange(0, vmax)) # 
    v_map_1_transpose = v_map_1.T # (1030, 1)
    umax_one = np.mat(np.ones(umax)).astype(int) # (1, 1720)
    v_map = v_map_1_transpose * umax_one # (1030, 1720)
    vmax_one = np.mat(np.ones(vmax)).astype(int)
    vmax_one_transpose = vmax_one.T # (1030, 1)
    u_map_1 = np.mat(np.arange(0, umax)) # (1, 1720)
    u_map = vmax_one_transpose * u_map_1 # (1030, 1720)
    Su = np.sum(u)
    Sv = np.sum(v)
    Sd = np.sum(d)
    Su2 = np.sum(np.square(u))
    Sv2 = np.sum(np.square(v))
    Sdu = np.sum(np.multiply(u, d))
    Sdv = np.sum(np.multiply(v, d))
    Suv = np.sum(np.multiply(u, v))
    n= len(u)
    beta0 = (np.square(Sd) * (Sv2 + Su2) - 2 * Sd * (Sv * Sdv + Su * Sdu) + n * (np.square(Sdv) + np.square(Sdu)))/2
    beta1 = (np.square(Sd) * (Sv2-Su2) + 2 * Sd * (Su*Sdu-Sv*Sdv) + n * (np.square(Sdv) - np.square(Sdu)))/2
    beta2 = -np.square(Sd) * Suv + Sd * (Sv * Sdu + Su * Sdv) - n * Sdv * Sdu
    gamma0 = (n * Sv2 + n * Su2 - np.square(Sv) - np.square(Su))/2
    gamma1 = (n * Sv2 - n * Su2 - np.square(Sv) + np.square(Su))/2
    gamma2 = Sv * Su - n * Suv
    A = (beta1 * gamma0 - beta0 * gamma1)
    B = (beta0 * gamma2 - beta2 * gamma0)
    C = (beta1 * gamma2 - beta2 * gamma1)
    delta = np.square(A) + np.square(B) - np.square(C)
    tmp1 = (A + np.sqrt(delta))/(B-C)
    tmp2 = (A - np.sqrt(delta))/(B-C)
    theta1 = math.atan(tmp1)
    theta2 = math.atan(tmp2)
    u=np.mat(u)
    v=np.mat(v)
    d=np.mat(d)
    d=d.T
    u=u.T
    v=v.T
    t1 = v * math.cos(theta1) - u * math.sin(theta1)
    t2 = v * math.cos(theta2) - u * math.sin(theta2)
    n_ones = np.ones(n).astype(int)
    n_ones = (np.mat(n_ones)).T
    T1 = np.hstack((n_ones, t1))
    T2 = np.hstack((n_ones, t2))
    f1 = d.T * T1 * np.linalg.inv (T1.T * T1) * T1.T * d
    f2 = d.T * T2 * np.linalg.inv (T2.T * T2) * T2.T * d
    if f1 < f2:
        theta = theta2
    else:
        theta = theta1
    t = v * math.cos(theta) - u * math.sin(theta)
    T = np.hstack((n_ones, t))
    a = np.linalg.inv(T.T * T) * T.T * d
    t_map = v_map * math.cos(theta) - u_map * math.sin(theta)
    newdisp = (a[0] + np.multiply(a[1], t_map))# - 20
    # exit()
    return newdisp

def get_ncc_sim(left, right, ncc_rad = 7, max_disp = None):
    _,C,H,W = left.shape
    if max_disp is None:
        max_disp = W

    Conf = torch.zeros(H,W,max_disp).cuda()-1
    N = (ncc_rad*2+1)*(ncc_rad*2+1)

    # ncc initial 
    # ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    
    
    # unfold
    left_unfold = ncc_Unfold(left_padded)
    left_unfold = left_unfold.view(1,C,N,H,W).permute(0,1,3,4,2).squeeze(0) # [C,H,W,rad^]
    right_unfold = ncc_Unfold(right_padded)
    right_unfold = right_unfold.view(1,C,N,H,W).permute(0,1,3,4,2).squeeze(0) # [C,H,W,rad^]
    
    # matmul = torch.sum(left_unfold*right_unfold,dim=-1) # [C,H,W]
    left_mean = torch.mean(left_unfold,dim=-1) # [C,H,W]
    right_mean = torch.mean(right_unfold,dim=-1) # [C,H,W]
    left_std = torch.std(left_unfold,dim=-1)+0.1 # [C,H,W]
    right_std = torch.std(right_unfold,dim=-1)+0.1 # [C,H,W]
    
    for i in range(max_disp):
        if i > 0:
            Conf[:, i:, i] = ((1/(N*left_std[:, :, i:]*right_std[:, :, :-i]))*(torch.sum(left_unfold[:, :, i:,:]*right_unfold[:, :, :-i,:],dim=-1)-N*left_mean[:, :, i:]*right_mean[:, :, :-i])).mean(dim=0)
            
        else:
            Conf[:, :, i] = ((1/(N*left_std*right_std))*(torch.sum(left_unfold*right_unfold,dim=-1)-N*left_mean*right_mean)).mean(dim=0)
    return Conf

def writePFM(file, image, scale=1):
    file = open(file, 'wb')
 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    image = np.flipud(image)
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
 
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n'.encode() % scale)
 
    image.tofile(file)

def save_disp_results(disp_dir,png_dir,result,max_disp=None,display=True):
    if display:
        print('save in {} and {}, shape = {}'.format(png_dir,disp_dir,result.shape))
    if 'npy' in disp_dir:
        np.save(disp_dir, result)
    elif 'pfm' in disp_dir:
        writePFM(disp_dir,result.astype(np.float32))
    if png_dir is not None:
        disp_vis(png_dir,result,max_disp)

def sim_down_size(sim,down_size=1):
    img_x = sim.shape[-1]
    max_disp = int(img_x/down_size)
    # max_disp = img_x
    sim = sim[:,:,:max_disp]
    return sim

def sim_restore(sim,value=-1):
    H,W,D = sim.shape
    if W==D:
        return sim
    expand = torch.zeros((H,W,W-D),device='cuda')-1
    expand = expand+value
    sim = torch.cat((sim,expand),-1)
    return sim    

def results_decouple(results,img_H,img_W,reference_dir):
    mkpts0 = results['points_A'].T # [n,2] numpy
    mkpts1 = results['points_B'].T
    seed_disp = results['disp']

    seed_disp = seed_disp.detach().cpu().numpy()
    key_points_coordinate = np.concatenate((mkpts0,mkpts1),axis=1)
    img1_seed,img2_seed = seed_record(img_W,img_H,key_points_coordinate) # [H,W]
    seed_disp = reso_recover(seed_disp,reference_dir)

    return seed_disp,img1_seed
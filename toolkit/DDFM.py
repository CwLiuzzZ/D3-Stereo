import torch

import sys
sys.path.append('..')
from DFM.DeepFeatureMatcher import match_inference

@torch.no_grad()
def evaluate_Unimatch(data,model,device,args):
    len_ = 3
    
    args.RBF_cycle = [11,15,15]
    if args.ratio_th_start == 0:
        args.ratio_th = [0.65,0.65,0.78]
    else:
        list_ = []
        for i in range(len_-1):
            list_.append(args.ratio_th_start-0.1)
        list_.append(args.ratio_th_start)
        args.ratio_th = list_
    args.BF_i = 7
    args.pt_offset = 40

    with torch.no_grad():
        # MiddEval3
        results = match_inference(args,data['left'].to(device), data['right'].to(device), data['left_dir'][0],data['right_dir'][0],model.DDFM,layer_size=[2,4,8],bidirectional=True,padding=8,display=False) #[11,15,15]
    return results

@torch.no_grad()
def evaluate_LacGwc(data,model,device,args):
    len_ = 2
    # print('processing image {} and {}'.format(data['left_dir'][0],data['right_dir'][0]))
    
    args.RBF_cycle = [9,15]
    if args.ratio_th_start == 0:
        args.ratio_th = [0.65,0.7]
    else:
        list_ = []
        for i in range(len_-1):
            list_.append(args.ratio_th_start-0.1)
        list_.append(args.ratio_th_start)
        args.ratio_th = list_
    args.BF_i = 7
    args.pt_offset = 40


    with torch.no_grad():
        results = match_inference(args,data['left'].to(device), data['right'].to(device), data['left_dir'][0],data['right_dir'][0],model.module.DDFM,layer_size=[2,4],bidirectional=True,padding=12,display=False)
    return results

@torch.no_grad()
def evaluate_BGNet(data,model,device,args):
    len_ = 3
    # print('processing image {} and {}'.format(data['left_dir'][0],data['right_dir'][0]))     
    data['left'] = data['left'][0,0,:,:]*114/1000 + data['left'][0,1,:,:]*587/1000 + data['left'][0,2,:,:]*299/1000
    data['right'] = data['right'][0,0,:,:]*114/1000 + data['right'][0,1,:,:]*587/1000 + data['right'][0,2,:,:]*299/1000
    data['left'] = (data['left']*255).unsqueeze(0).unsqueeze(0)
    data['right'] = (data['right']*255).unsqueeze(0).unsqueeze(0)
    
    args.RBF_cycle = [11,15,15]
    if args.ratio_th_start == 0:
        args.ratio_th = [0.7,0.7,0.8]
    else:
        list_ = []
        for i in range(len_-1):
            list_.append(args.ratio_th_start-0.1)
        list_.append(args.ratio_th_start)
        args.ratio_th = list_
    args.BF_i = 7
    args.pt_offset = 30

        # args.RBF_cycle = [11,15,15,15]
    with torch.no_grad():
        results = match_inference(args,data['left'].to(device), data['right'].to(device), data['left_dir'][0],data['right_dir'][0],model.DDFM,layer_size=[2,4,8],bidirectional=True,padding=64,display=True) # 
    return results

def evaluate_graft(data,model,device,args):
    len_ = 3
    # print('processing image {} and {}'.format(data['left_dir'][0],data['right_dir'][0]))
    
    args.RBF_cycle = [9,12,15]
    if args.ratio_th_start == 0:
        args.ratio_th = [0.6,0.6,0.7]
    else:
        list_ = []
        for i in range(len_-1):
            list_.append(args.ratio_th_start-0.1)
        list_.append(args.ratio_th_start)
        args.ratio_th = list_

    model = graft_ddfm(model['fe'],model['adaptor'])

    args.BF_i = 5 
    with torch.no_grad():
        data['left'] = (data['left'].to(device))
        data['right'] = (data['right'].to(device))
        results = match_inference(args,data['left'].to(device), data['right'].to(device), data['left_dir'][0],data['right_dir'][0],model.DDFM,layer_size=[2,4,8],bidirectional=True,padding=16)
    return results

class graft_ddfm():
    def __init__(self,fe,adaptor):
        self.fe = fe
        self.adaptor = adaptor
    def DDFM(self,x):
        feature_list1,x = self.fe.module.DDFM(x)
        feature_list2 = self.adaptor.module.DDFM(x)
        return feature_list1+feature_list2
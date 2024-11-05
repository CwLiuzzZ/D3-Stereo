import torch

import sys
sys.path.append('..')
import toolkit.base_function as base_function


def evaluate_graft(data,model):

    imgL = data['left'].cuda()
    imgR = data['right'].cuda()

    padder = base_function.InputPadder(imgL.shape, divide = 16, mode = 'replicate') # [N,H,W]
    imgL, imgR = padder.pad(imgL, imgR)

    with torch.no_grad():
        # print(imgL.shape)
        
        left_fea = model['fe'](imgL) # 1/4 
        right_fea = model['fe'](imgR) # 1/4
        
        left_fea = model['adaptor'](left_fea) # 1/4 
        right_fea = model['adaptor'](right_fea) # 1/4
        
        pred_disp = model['agg_model'](left_fea, right_fea, imgL, training=False)
        
    pred_disp = padder.unpad(pred_disp.squeeze()) 
    pred_np = pred_disp.detach().cpu().numpy()
    return pred_np

@torch.no_grad()
def evaluate_Unimatch(data,model):
    image1 = data['left'].cuda()
    image2 = data['right'].cuda()
    
    # print('processing image {} and {}'.format(data['left_dir'][0],data['right_dir'][0]))
    
    padder = base_function.InputPadder(image1.shape, 32,'sintel') # 16 or 32?
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        pred_disp = model(image1, image2,
                            attn_type='self_swin2d_cross_swin1d',
                            attn_splits_list=[2,8],
                            corr_radius_list=[-1,4],
                            prop_radius_list=[-1,1],
                            num_reg_refine=3, # 1
                            task='stereo',
                            )['flow_preds'][-1]  # [1, H, W]

    # remove padding
    pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
    pred_disp = pred_disp.detach().cpu().numpy()

    return pred_disp

def evaluate_LacGwc(data,model):
    image1 = data['left'].cuda()
    image2 = data['right'].cuda()
    # print('processing image {} and {}'.format(data['left_dir'][0],data['right_dir'][0]))
        
    padder = base_function.InputPadder(image1.shape,16, mode = 'replicate')
    image1, image2 = padder.pad(image1, image2)
    with torch.no_grad():
        pred_disp,_,_ = model(image1, image2)
    pred_disp = padder.unpad(pred_disp)
    pred_disp = pred_disp.squeeze().detach().cpu().numpy()
    return pred_disp

def evaluate_BGNet(data,model):
    image1_dir = data['left_dir'][0]
    image2_dir = data['right_dir'][0]
    # print('processing image {} and {}'.format(image1_dir,image2_dir))
    
    image1 = data['left']
    image2 = data['right']
    image1 = image1[0,0,:,:]*114/1000 + image1[0,1,:,:]*587/1000 + image1[0,2,:,:]*299/1000
    image2 = image2[0,0,:,:]*114/1000 + image2[0,1,:,:]*587/1000 + image2[0,2,:,:]*299/1000
    image1 = (image1*255).unsqueeze(0).unsqueeze(0)
    image2 = (image2*255).unsqueeze(0).unsqueeze(0)
    # image1 = F.interpolate(image1, size=(512,1024), mode='bilinear', align_corners=True) 
    # image2 = F.interpolate(image2, size=(512,1024), mode='bilinear', align_corners=True) 

    padder = base_function.InputPadder(image1.shape,64, mode = 'replicate')
    image1, image2 = padder.pad(image1, image2)

    pred,_,_,_ = model(image1.cuda(), image2.cuda()) 
    pred = padder.unpad(pred)

    pred_disp = pred[0].data.cpu().numpy()
    return pred_disp

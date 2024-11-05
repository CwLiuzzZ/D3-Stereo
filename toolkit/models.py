import os
import torch

from unimatch.unimatch.unimatch import UniMatch
from LacGwcNet.networks.stackhourglass import PSMNet as PSMNet_LacGwc
from BGNet.models.bgnet_plus import BGNet_Plus
from Graft_PSMNet.networks import Aggregator as Agg
from Graft_PSMNet.networks import U_net as un
from Graft_PSMNet.networks import feature_extraction as FE
import toolkit.evaluate as evaluate_func
import toolkit.DDFM as DDFM
from collections import OrderedDict


def prapare_main_function(hparams,device = 'cuda',pre_trained=True):
    inference_type = hparams.inference_type
    if hparams.network == 'graft':
        model_dir = "models/graft/checkpoint_final_10epoch.tar"
        model = load_GRAFT_model(hparams,model_dir)
        if inference_type == 'evaluate':
            inference = evaluate_func.evaluate_graft
        if inference_type == 'DDFM':
            inference = DDFM.evaluate_graft
    elif hparams.network == 'Unimatch':
        model_dir = 'models/Unimatch/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth'
        model = load_unimatch_model(hparams,model_dir,device)
        if inference_type == 'evaluate':
            inference = evaluate_func.evaluate_Unimatch
        elif inference_type == 'DDFM':
            inference = DDFM.evaluate_Unimatch
    elif hparams.network == 'LacGwc':
        model_dir = 'models/LacGwc/kitti2015.pth'
        model = load_LacGwc_model(hparams,model_dir,device)
        if inference_type == 'evaluate':
            inference = evaluate_func.evaluate_LacGwc
        elif inference_type == 'DDFM':
            inference = DDFM.evaluate_LacGwc
    elif hparams.network == 'BGNet':
        # if pre_trained:
        model_dir = 'models/BGNet/kitti_15_BGNet_Plus.pth'
        model = load_BGNet_model(model_dir)
        if inference_type == 'evaluate':
            inference = evaluate_func.evaluate_BGNet
        elif inference_type == 'DDFM':
            inference = DDFM.evaluate_BGNet
    return model,inference,inference_type,hparams

def load_GRAFT_model(args,model_dir):
    
    fe_model = FE.VGG_Feature(fixed_param=True).eval()
    adaptor = un.U_Net_v4(img_ch=256, output_ch=64).eval()
   
    agg_model = Agg.PSMAggregator(192, udc=True).eval()

    if 'tar' in model_dir:
        fe_model = torch.nn.DataParallel(fe_model.cuda())
        adaptor = torch.nn.DataParallel(adaptor.cuda())
        agg_model = torch.nn.DataParallel(agg_model.cuda())
        if os.path.exists(model_dir):
            print('load pretrained model from {}'.format(model_dir))
            pretrain_dict = torch.load(model_dir, map_location=torch.device('cpu'))
            adaptor.load_state_dict(pretrain_dict['fa_net'])
            agg_model.load_state_dict(pretrain_dict['net'])
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    elif 'ckpt' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained graft:', model_dir)
            pretrain_dict = torch.load(model_dir, map_location=torch.device('cpu'))
            msg = adaptor.load_state_dict(pretrain_dict['fa_net'],strict=True)
            msg = agg_model.load_state_dict(pretrain_dict['net'],strict=True)
            fe_model = torch.nn.DataParallel(fe_model.cuda())
            adaptor = torch.nn.DataParallel(adaptor.cuda())
            agg_model = torch.nn.DataParallel(agg_model.cuda())
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
        
    return {'fe':fe_model,'adaptor':adaptor,'agg_model':agg_model}

def load_unimatch_model(args,model_dir,device):
    # model
    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     num_head=1,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task='stereo').to(device)

    if 'pth' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained Unimatch:', model_dir)
            model.load_state_dict(torch.load(model_dir)['model'], strict=False)
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    elif 'ckpt' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained Unimatch:', model_dir)
            load_pretrained_net(model, model_dir, strict=True,print_msg=True)
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    return model

def load_LacGwc_model(args,model_dir,device):

    affinity_settings = {}
    affinity_settings['win_w'] = 3
    affinity_settings['win_h'] = 3
    affinity_settings['dilation'] = [1, 2, 4, 8]
    udc = not False

    # model
    model = PSMNet_LacGwc(maxdisp=192, struct_fea_c=4, fuse_mode='separate',
               affinity_settings=affinity_settings, udc=udc, refine='csr')
    if 'pth' in model_dir:
        model = torch.nn.DataParallel(model)
        if os.path.exists(model_dir):
            print('=> Loading pretrained LacGwc:', model_dir)
            model.load_state_dict(torch.load(model_dir))
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    elif 'ckpt' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained LacGwc:', model_dir)
            load_pretrained_net(model, model_dir, strict=True,print_msg=True)
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def load_BGNet_model(model_dir):
    # model
    model = BGNet_Plus()
    if 'pth' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained BGNet:', model_dir)
            model.load_state_dict(torch.load(model_dir,map_location=lambda storage, loc: storage))
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    elif 'ckpt' in model_dir:
        if os.path.exists(model_dir):
            print('=> Loading pretrained BGNet:', model_dir)
            load_pretrained_net(model, model_dir, strict=True,print_msg=True)
        else:
            print('{} does not exists => Using random initialization'.format(model_dir))
    return model
    
def load_pretrained_net(net, pretrained_path,strict=False,print_msg=False):
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location='cpu')
        weights = state['state_dict'] if 'state_dict' in state.keys() else state
        new_state_dict = OrderedDict()

        for k, v in weights.items():
            name = k[7:] if k.startswith('module.') else k
            name = k[6:] if k.startswith('model.') else k
            new_state_dict[name] = v

        msg = net.load_state_dict(new_state_dict, strict=strict) 
        if print_msg:
            print(msg)
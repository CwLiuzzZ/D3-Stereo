import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

import sys
sys.path.append('..')
from toolkit.data_loader.pre_post_process import results_post_process,generate_appended_file_dirs
from toolkit.base_function import save_disp_results,results_decouple
from toolkit.backup.dataset_function import generate_file_lists
from toolkit.data_loader.dataloader import prepare_dataset,dataloader_customization
from models import prapare_main_function
from toolkit.args.args_default import get_opts
import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
# ignore tensorboard warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
torch.backends.cudnn.benchmark = True


write_dict = {}

def inference(hparams):

    ###################### 
    # prepare dataloader # 
    ###################### 
    hparams,aug_config = dataloader_customization(hparams)

    file_path_dic = generate_file_lists(dataset = hparams.dataset,if_train=hparams.dataset_type=='train',method='gt',save_method=hparams.save_name)
    dataset,n_img = prepare_dataset(file_path_dic,aug_config=aug_config)
    hparams.num_steps = int(n_img*hparams.epoch_size/hparams.batch_size)
    disp_list_noc,disp_list_fd = generate_appended_file_dirs(hparams.dataset,n_img)

    ############################################
    # load model and select inference function #
    ############################################
    model,inference,inference_type,hparams = prapare_main_function(hparams)
    
    ##########################
    # run inference function #
    ##########################
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')
    dataloader = DataLoader(dataset, batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)
    if inference_type == 'DDFM':
        if model is not None:
            if isinstance(model,dict):
                for i in model:
                    model[i].cuda()
                    model[i].eval()
            else:
                model.cuda()
                model.eval()
        for (i,data) in tqdm(enumerate(dataloader),desc = "evaluate {} for {}".format(hparams.network,hparams.dataset),total=len(dataloader)):
            img_H,img_W = data['left'].shape[-2],data['left'].shape[-1]
            results = inference(data,model,'cuda',hparams)
            seed_disp,img1_seed = results_decouple(results,img_H,img_W,data['left_dir'][0])
            pred_disp = seed_disp # use dense results
            # pred_disp = img1_seed # use sparse results
            pred_disp = results_post_process(pred_disp,hparams.dataset,hparams.dataset_type,data,noc_dir=disp_list_noc[i],pt_fd_dir=disp_list_fd[i])
            save_disp_results(data['save_dir_disp'][0],data['save_dir_disp_vis'][0],pred_disp,hparams.max_disp,display=False)
            torch.cuda.empty_cache()
    elif inference_type == 'evaluate':
        if model is not None:
            if isinstance(model,dict):
                for i in model:
                    model[i].cuda()
                    model[i].eval()
            else:
                model.cuda()
                model.eval()
        for (i,data) in tqdm(enumerate(dataloader),desc = "evaluate {} for {}".format(hparams.network,hparams.dataset),total=len(dataloader)):
            pred_disp = inference(data,model)
            pred_disp = results_post_process(pred_disp,hparams.dataset,hparams.dataset_type,data,noc_dir=disp_list_noc[i],pt_fd_dir=disp_list_fd[i])
            save_disp_results(data['save_dir_disp'][0],data['save_dir_disp_vis'][0],pred_disp,hparams.max_disp,display=False)
            torch.cuda.empty_cache()

def reconfig():
    hparams = get_opts()
    hparams.inference_type = 'DDFM' # DDFM, evaluate
    hparams.num_workers = 1
    hparams.batch_size = 1
    hparams.pt = True
    return hparams

if __name__ == '__main__':       
    for dataset in ['realroad']:
    # for dataset in ['realroad','VirtualRoad']:
        for network in ['graft','BGNet','Unimatch','LacGwc']:
            hparams = reconfig()
            hparams.network = network
            hparams.dataset = dataset
            hparams.save_name = network+'_D3Stereo'
            inference(hparams)
    
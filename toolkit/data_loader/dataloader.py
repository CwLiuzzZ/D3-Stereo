import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

import sys
sys.path.append('../..')
from toolkit.base_function import io_disp_read
from toolkit.data_loader import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

aug_config_dic = {
    'Unimatch':{'resize':None,'RandomColor':False,'RandomHorizontalFlip':False,'VerticalFlip':False,'norm':True,'crop_size':None},
    'LacGwc':{'resize':None,'RandomColor':False,'RandomHorizontalFlip':False,'VerticalFlip':False,'norm':True,'crop_size':None},
    'BGNet':{'resize':None,'RandomColor':False,'RandomHorizontalFlip':False,'VerticalFlip':False,'norm':False,'crop_size':None},
    'graft':{'resize':None,'RandomColor':False,'RandomHorizontalFlip':False,'VerticalFlip':False,'norm':True,'crop_size':None},
    'delete':{'resize':None,'RandomColor':False,'RandomHorizontalFlip':False,'VerticalFlip':False,'norm':False,'crop_size':None}}

def dataloader_customization(hparams):
    
    if hparams.save_name == 'Delete':
        hparams.save_name = hparams.network
    aug_config = aug_config_dic[hparams.network]

    if hparams.keep_size:
        aug_config['resize']=None
    
    if 'VirtualRoad' in hparams.dataset:
        hparams.max_disp = 200
        if 'pt' in hparams.dataset:
            hparams.max_disp_sr = 60
        else:
            hparams.max_disp_sr = 200
    if 'realroad' in hparams.dataset:
        hparams.max_disp = 200
        if 'pt' in hparams.dataset:
            hparams.max_disp_sr = 80
        else:
            hparams.max_disp_sr = 250

    return hparams,aug_config


def prepare_dataset(file_paths_dic, aug_config):
    '''
    function: make dataloader
    input:
        file_paths_dic: store file paths
        aug_config: configuration for augment
    output:
        dataloader
    '''
    # augmentation
    train_transform_list = []
    if not aug_config['resize'] is None:
        train_transform_list.append(transforms.ResizeImage(aug_config['resize']))
    if aug_config['RandomColor']:
        train_transform_list.append(transforms.RandomColor())
    if aug_config['VerticalFlip']:
        train_transform_list.append(transforms.RandomVerticalFlip())
    if aug_config['RandomHorizontalFlip']:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    train_transform_list.append(transforms.ToTensor())
    # if 'gray' in aug_config.keys():
    #     if aug_config['gray']:
    #         train_transform_list.append(transforms.Gray())
    if aug_config['norm']:
        train_transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    train_transform = transforms.Compose(train_transform_list)
    if not aug_config['crop_size'] is None:
        crop_size = aug_config['crop_size']
    else:
        crop_size = None
    
    dataset = StereoDataset(file_paths_dic,crop_size,transform=train_transform)
    
    n_img = len(dataset)
    print('Use a dataset with {} samples'.format(n_img))
    return dataset,n_img

def default_loader(path):
    '''
        function: read left and right images
        output: array
    '''
    img_BGR = cv2.imread(path)
    img_rgb= cv2.cvtColor(img_BGR,cv2.COLOR_BGR2RGB)
    
    return img_rgb

def spatial_transform(img1, img2, disp,crop_size=(600,2000)):
    '''
    function: crop images and disparity maps
    '''

    assert img1.shape[0] >= crop_size[0], str(img1.shape) +' '+ str(crop_size)
    assert img1.shape[1] >= crop_size[1], str(img1.shape) +' '+ str(crop_size)

    # if img1.shape[0] < crop_size[0]:
    #     crop_size = (img1.shape[0],crop_size[1])
    
    # if img1.shape[1] < crop_size[1]:
    #     crop_size = (crop_size[1][0],img1.shape[1])

    if img1.shape[0] == crop_size[0]:
        y0 = img1.shape[0]
    else:
        y0 = np.random.randint(0, img1.shape[0] - crop_size[0])
    if img1.shape[1] == crop_size[1]:
        x0 = img1.shape[1]
    else:
        x0 = np.random.randint(0, img1.shape[1] - crop_size[1])
    
    y0 = np.clip(y0, 0, img1.shape[0] - crop_size[0])
    x0 = np.clip(x0, 0, img1.shape[1] - crop_size[1])


    img1 = img1[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    img2 = img2[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    disp = disp[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    
    img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    disp = cv2.resize(disp/2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    return img1,img2,disp

class StereoDataset(Dataset):
    def __init__(self, file_paths_dic,crop_size,transform, loader=default_loader, dploader=io_disp_read):
        super(StereoDataset, self).__init__()
        self.transform = transform
        self.loader = loader
        self.disploader = dploader
        self.samples = []
        self.crop_size = crop_size
        self.load_disp = True

        self.lefts = file_paths_dic['left_list']
        self.rights = file_paths_dic['right_list']
        self.disps = file_paths_dic['disp_list']
        self.save_dirs1 = file_paths_dic['save_path_disp']
        self.save_dirs2 = file_paths_dic['save_path_disp_image']
        
        # print('number of files: left image {}, right image {}, disp {}'.format(len(self.lefts), len(self.rights), len(self.disps)))
        assert len(self.lefts) == len(self.rights), "{},{}".format(len(self.lefts),len(self.rights))
        if not len(self.disps) == len(self.lefts):
            self.load_disp = False
        # assert len(self.lefts) == len(self.rights) == len(self.disps), "{},{},{}".format(len(self.lefts),len(self.rights),len(self.disps))
        for i in range(len(self.lefts)):
            sample = dict()
            sample['left'] = self.lefts[i]
            sample['right'] = self.rights[i]
            if self.load_disp:
                sample['disp'] = self.disps[i]
            sample['save_dir1'] = self.save_dirs1[i]
            sample['save_dir2'] = self.save_dirs2[i]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        sample = {}
        sample_path = self.samples[index]
    
        sample['left'] = self.loader(sample_path['left']) # array
        sample['right'] = self.loader(sample_path['right']) # array
        if self.load_disp:
            sample['disp'] = self.disploader(sample_path['disp'])
            sample['disp_dir'] = sample_path['disp']
        else:
            sample['disp'] = np.zeros(shape=(sample['left'].shape[0],sample['left'].shape[1]))
        sample['left_dir'] = sample_path['left']
        sample['right_dir'] = sample_path['right']
        sample['save_dir_disp'] = sample_path['save_dir1']
        sample['save_dir_disp_vis'] = sample_path['save_dir2']
        sample['ori_shape'] = sample['disp'].shape

        if not self.crop_size is None:
            sample['left'],sample['right'],sample['disp'] = spatial_transform(sample['left'],sample['right'],sample['disp'],crop_size=self.crop_size)
        sample['left']=Image.fromarray(sample['left'])
        sample['right']=Image.fromarray(sample['right'])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

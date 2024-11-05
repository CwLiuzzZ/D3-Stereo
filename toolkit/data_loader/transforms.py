from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class Gray(object):
    def __call__(self, sample):
        norm_keys = ['left', 'right']
        for key in norm_keys:
            tensor = sample[key]
            assert len(tensor.shape) == 3, str(tensor.shape)
            R = tensor[0]
            G = tensor[1]
            B = tensor[2]
            tensor=0.299*R+0.587*G+0.114*B
            sample[key] = tensor.unsqueeze(0)
        return sample

class ResizeImage(object):
    def __init__(self, size=(256, 512)):
        self.size=size
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        left_image = sample['left']
        right_image = sample['right']
        disp_image = sample['disp']
        sample['disp'] = sample['disp']*self.size[1]/sample['ori_shape'][-1]
        disp_image = Image.fromarray(sample['disp'])
        new_left_image = self.transform(left_image)
        new_right_image = self.transform(right_image)
        new_disp_image = self.transform(disp_image)
        sample['left'] = new_left_image
        sample['right'] = new_right_image
        sample['disp'] = new_disp_image
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __init__(self, tensor_norm=255.):
        self.tensor_norm = tensor_norm

    def __call__(self, sample):
        sample = ToNumpyArray()(sample)
        left = sample['left']
        right = sample['right']
        assert len(left.shape) in [2,3], str(left.shape)
        if len(left.shape) == 2:
            sample['left'] = torch.from_numpy(left).unsqueeze(0) / self.tensor_norm
            sample['right'] = torch.from_numpy(right).unsqueeze(0) / self.tensor_norm
        elif len(left.shape) == 3:
            left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
            right = np.transpose(sample['right'], (2, 0, 1))
            sample['left'] = torch.from_numpy(left)/ self.tensor_norm
            sample['right'] = torch.from_numpy(right)/ self.tensor_norm

        disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        disp = sample['disp']  # [H, W]
        sample['disp'] = torch.from_numpy(disp)
        return sample


# 经过 toTensor变为(0,1)， 再经过normalize变为(-1,1)
class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['left'] = np.lib.pad(sample['left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['right'] = np.lib.pad(sample['right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            sample['disp'] = np.lib.pad(sample['disp'],
                                        ((top_pad, 0), (0, right_pad)),
                                        mode='constant',
                                        constant_values=0)
        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            sample['disp'] = self.crop_img(sample['disp'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]
        
class RandomHorizontalFlip(object):
    def __init__(self):
        self.transform = transforms.RandomHorizontalFlip(p=1)

    def __call__(self, sample):
        left_image = sample['left']
        right_image = sample['right']
        disp_image = sample['disp']
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            fliped_left = self.transform(right_image)
            fliped_right = self.transform(left_image)
            fliped_disp = self.transform(disp_image)
            sample['left'] = fliped_left
            sample['right'] = fliped_right
            sample['disp'] = fliped_disp
        return sample

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))
            sample['disp'] = np.copy(np.flipud(sample['disp']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)
        sample['disp'] = np.array(sample['disp']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)
            sample['right'] = F.adjust_gamma(sample['right'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left'] = F.adjust_brightness(sample['left'], brightness)
            sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)
            sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['left'] = F.adjust_saturation(sample['left'], saturation)
            sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        # sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)
        return sample

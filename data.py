import os
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import numpy

import argparse
import parser

import torchvision.transforms.functional as TF
import random


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode

        if self.mode == 'train' or self.mode == 'val':
            self.data_dir = os.path.join(args.data_dir, mode)
        else:
            self.data_dir = args.data_dir 
        
        # append the image and seg directories
        self.img_dir = os.path.join(self.data_dir, 'img')
        self.seg_dir = os.path.join(self.data_dir, 'seg')

        # Filenames of all files in img and seg directories, as a list
        img_names = sorted(os.listdir(self.img_dir))
        seg_names = sorted(os.listdir(self.seg_dir))

        # self.data is a list with length equal to the number of images
        # Each element is another list, with 1st element being the filepath to that image, and 2nd being the filepath to the segmentation map of that image
        i = 0
        self.data = []
        for i in range(len(img_names)):
            self.data.append( [os.path.join(self.img_dir, img_names[i]), os.path.join(self.seg_dir, seg_names[i])] ) 
            i += 1
            
        # Transform the image (not segmentation map)
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.data)

    def custom_transform(self, image, segmentation):
        # 0.5 probability of performing one transformation
        if random.random() > 0.5:
            p = random.random()

            # Rotate
            if p < 0.3:
                angle = random.randint(-30, 30)
                image = TF.rotate(image, angle)
                segmentation = TF.rotate(segmentation, angle)
            
            # Horizontal flip
            elif p < 0.6:
                TF.hflip(image)
                TF.hflip(segmentation)
            
            # Vertical flip
            elif p < 0.9:
                TF.vflip(image)
                TF.vflip(segmentation)
            
            # Colours
            else:
                TF.adjust_brightness(image, 1.4)
                TF.adjust_hue(image, 0.3)
                TF.adjust_contrast(image, 1.3)
                TF.adjust_saturation(image, 0.7)
        
        return image, segmentation

    def __getitem__(self, idx):

        ''' get data '''
        img_path, seg_path = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path).convert('L') # Load as grayscale Image object

        # Only transform image+seg map if it's the training set. This only needs to be commented out when TRAINING the baseline model. Testing is unaffected
        """
        if self.mode == "train":
            img, seg = self.custom_transform(img, seg)
        """
        seg = numpy.array(seg) # convert to numpy array
        seg = torch.from_numpy(seg) # convert to byte tensor
        seg = seg.long() # convert to long tensor

        return self.transform(img), seg
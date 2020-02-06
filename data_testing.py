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
        self.img_dir = args.data_dir 

        # Filenames of all files in img and seg directories, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # self.data is a list with length equal to the number of images
        # Each element is the filepath to an image
        self.data = []
        for i in range(len(img_names)):
            self.data.append( os.path.join(self.img_dir, img_names[i]) ) 
            
        # Transform the image (not segmentation map)
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img)
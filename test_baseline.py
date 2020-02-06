import os
import torch

import parser
import models_baseline
import data_testing

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

from mean_iou_evaluate import mean_iou_score
from PIL import Image

if __name__ == '__main__':
    
    # run baseline model on the test set and save predicted segmentation maps to a directory
    print('*****Testing baseline model*****')
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data_testing.DATA(args, mode='test'),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare model '''
    model = models_baseline.Net(args).cuda()

    ''' resume saved model '''
    print('===> resume saved model ...')
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    # Save predicted segmentation maps
    print('===> calculate predictions ...')
    model.eval()
    preds = []
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, imgs in enumerate(test_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            
            _, pred = torch.max(pred, dim = 1)
            pred = pred.cpu().numpy().squeeze() # numpy.ndarray 45x352x448, 5x352x448 on last one
            preds.append(pred)

    preds = np.concatenate(preds) # numpy.ndarray 500x352x448, signed 8 byte integers

    print('===> save predictions ...')
    # create prediction directory if it doesn't exist
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    
    # separate the images
    for i in range(500):
        seg_map = Image.fromarray( preds[i][:][:].astype('uint8') )
        if i < 10:
            seg_map.save( os.path.join(args.pred_dir, "000{}.png".format(i)) )
        elif i < 100:
            seg_map.save( os.path.join(args.pred_dir, "00{}.png".format(i)) )
        else:
            seg_map.save( os.path.join(args.pred_dir, "0{}.png".format(i)) )

    print('*****Testing Complete*****')

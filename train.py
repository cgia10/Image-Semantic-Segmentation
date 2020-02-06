import argparse
import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    

if __name__=='__main__':
    
    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    
    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    ''' load model '''
    print('===> prepare model ...')
    model = models.Net(args)
    model.cuda() # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        
        model.train()
        
        for idx, (imgs, cls) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, cls = imgs.cuda(), cls.cuda()
            
            ''' forward path '''
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, cls) # compute loss
            
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss vs iters', loss.data.cpu().numpy(), iters) # training loss vs num of iterations
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = test.evaluate(model, val_loader, 0)        
            writer.add_scalar('val_acc vs epoch', acc, epoch) # mIOU score on validation set vs num of epochs
            writer.add_scalar('val_acc vs iters', acc, iters) # mIOU score on validation set vs num of iterations
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
            
            ''' save best model '''
            if acc > best_acc:
                if args.model_level == 'baseline':
                    save_model(model, os.path.join(args.save_dir, 'baseline_model.pth.tar'))
                else:
                    save_model(model, os.path.join(args.save_dir, 'best_model.pth.tar'))
                best_acc = acc
                best_epoch = epoch

    print("********** FINISHED TRAINING **********")
    print("Best accuracy = {} (epoch #{})".format(best_acc, best_epoch))

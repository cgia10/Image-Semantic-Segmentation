from __future__ import absolute_import
import argparse


def arg_parse():
    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='./data/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=1, type=int,
                    help="num of epochs")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=45, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=45, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial weight decay")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default="./log/baseline_model.pth.tar", 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='./log/') # directory for trained model and tensorboard info
    parser.add_argument('--pred_dir', type=str, default='./predictions/') # directory for predicted segmentation maps
    parser.add_argument('--model_level', type=str, default='baseline') # save the name of the trained model as baseline or improved
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args

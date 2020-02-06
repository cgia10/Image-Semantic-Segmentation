import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
import parser
import torchvision.transforms as transforms

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  

        ''' declare layers used in this network'''
        # Resnet block
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-2]
        resnet18 = nn.Sequential(*modules)
        self.resnet18 = resnet18 # 3x352x448 -> 512x11x14
        
        # first block
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False) # 512x11x14 -> 256x22x28
        self.d1 = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        
        # second block
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False) # 256x22x28 -> 128x44x56
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        # third block
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False) # 128x44x56 -> 64x88x112
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # fourth block
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False) # 64x88x112 -> 32x176x224
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        # fifth block
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False) # 32x176x224 -> 16x352x448
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()

        # sixth block
        self.conv6 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True) # 16x352x448 -> 9x352x448


    def forward(self, img):

        x = self.resnet18(img)
        #print("After resnet layer: ", x.size())
        
        x = self.relu1( self.bn1( self.d1( self.conv1(x) ) ) )
        #print("After 1st conv layer: ", x.size())
        
        x = self.relu2( self.bn2( self.conv2(x) ) )
        #print("After 2nd conv layer: ", x.size())

        x = self.relu3( self.bn3( self.conv3(x) ) )
        #print("After 3rd conv layer: ", x.size())

        x = self.relu4( self.bn4( self.conv4(x) ) )
        #print("After 4th conv layer: ", x.size())

        x = self.relu5( self.bn5( self.conv5(x) ) )
        #print("After 5th conv layer: ", x.size())

        x = self.conv6(x)
        #print("After 6th conv layer: ", x.size())

        return x
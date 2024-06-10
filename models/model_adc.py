# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1x1 convolution
        
        #3x3 + 3x3 convolution (5x5 convolution)--> 1 block
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, 
                                kernel_size=3, padding = 1)
        self.conv1_2 =  nn.Conv2d(in_channels=16, out_channels=32, 
                                kernel_size=3, padding = 1)
        self.ReLU = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        
        # 3x3 convolution --> 2 block
        self.conv1x1_2 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=48, 
                                kernel_size=3, padding = 1)
        self.batch2 = nn.BatchNorm2d(48)
        
        # max pooling
        
        # 3x3 convolution --> 3 block
        self.conv1x1_3 = nn.Conv2d(in_channels = 48, out_channels = 16, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, 
                                kernel_size=3, padding = 1)

        self.batch3 = nn.BatchNorm2d(64)
        
       
        # Fully connected layers + dropout
        self.drop = nn.Dropout(0.8)
        self.fc1 = nn.Linear(in_features=22*22*64, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features = 50, out_features = 2)
        
        #torch.save(self.state_dict(),PATH)
    

    def forward(self, x):
        # First block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)
        x = self.batch1(x)
        
        # Second block
        x = self.conv1x1_2(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.batch2(x)
        
        x = F.max_pool2d(x, 2, 2)
        
        # Third block
        x = self.conv1x1_3(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.batch3(x)

        
        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        
        return x
    
    
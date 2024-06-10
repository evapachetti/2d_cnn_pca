# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        print('l: ',l.size())
        print('g: ', g.size())
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g
    
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class Net(nn.Module):
    def __init__(self, im_size, attention = True, normalize_attn=True):
        super(Net, self).__init__()
        # 1x1 convolution
        self.attention = attention
        
        self.dense = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=1, padding=0, bias=True)

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
        
        # 4 block
        self.conv1x1_4 = nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=80, 
                                kernel_size=3, padding = 1)
        self.batch4 = nn.BatchNorm2d(80)

        
        # Fully connected layers + dropout
        self.drop = nn.Dropout(0.8)
        self.fc1 = nn.Linear(in_features=16*16*80, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features = 50, out_features = 2)
        
        # Attention gates
        if self.attention:
            self.projector1 = ProjectorBlock(48, 80)
            self.attn1 = LinearAttentionBlock(in_features=80, normalize_attn=normalize_attn)
            self.projector2 = ProjectorBlock(64, 80)
            self.attn2 = LinearAttentionBlock(in_features=80, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=80, normalize_attn=normalize_attn)
            
        if self.attention:
            self.classify = nn.Sequential(
                nn.Linear(in_features=64*64*80*3, out_features=500),
                nn.ReLU(),
                nn.Dropout(0.8),
                nn.Linear(in_features=500, out_features=50),
                nn.ReLU(),
                nn.Dropout(0.8),
                nn.Linear(in_features = 50, out_features = 2),
                nn.Dropout(0.8))
        else:
            self.classify = nn.Sequential(
                nn.Linear(in_features=64*64*80, out_features=500),
                nn.ReLU(),
                nn.Dropout(0.8),
                nn.Linear(in_features=500, out_features=50),
                nn.ReLU(),
                nn.Dropout(0.8),
                nn.Linear(in_features = 50, out_features = 2),
                nn.Dropout(0.8))
            

    def forward(self, x):
        # First block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)
        x = self.batch1(x)
        
        # Second block
        x = self.conv1x1_2(x)
        x = self.ReLU(x)
        l2 = self.conv2(x)
        x = self.ReLU(l2)
        x = self.batch2(x)
                
        # Third block
        x = self.conv1x1_3(x)
        x = self.ReLU(x)
        l3 = self.conv3(x)
        x = self.ReLU(l3)
        x = self.batch3(x)
        
        # Forth block
        
        x = self.conv1x1_4(x)
        x = self.ReLU(x)
        l4 = self.conv4(x)
        x = self.ReLU(l4)
        x = self.batch4(x)
                
        g = self.dense(x)

        
        if self.attention:
            c1, g1 = self.attn1(self.projector1(l2), g)
            c2, g2 = self.attn2(self.projector2(l3), g)
            c3, g3 = self.attn3(l4, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]

    
    
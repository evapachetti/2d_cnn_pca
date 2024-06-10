# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.ReLU = nn.ReLU()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        self.branch1 = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            nn.MaxPool2d(2, 2),
            conv_block(32, 16, kernel_size=1, padding=0),
            conv_block(16, 48),
            nn.MaxPool2d(2, 2),
            conv_block(48, 16, kernel_size=1, padding=0),
            conv_block(16, 64),
            conv_block(64, 16, kernel_size=1, padding=0),
            conv_block(16, 80),
            nn.MaxPool2d(2, 2),
            conv_block(80, 16, kernel_size=1, padding=0),
            conv_block(16, 96),
        )
        
        self.branch2 = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            nn.MaxPool2d(2, 2),
            conv_block(32, 16, kernel_size=1, padding=0),
            conv_block(16, 48),
            nn.MaxPool2d(2, 2),
            conv_block(48, 16, kernel_size=1, padding=0),
            conv_block(16, 64),
        )
        
        self.fc1_t2 = nn.Linear(16 * 16 * 96, 500)
        self.fc1_adc = nn.Linear(22 * 22 * 64, 500)
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(50, 2)
        )
        
        self.fc1_comb = nn.Linear(4, 2)
    
    def forward(self, x1, x2):
        x1 = self.branch1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1_t2(x1)
        x1 = self.fc_layers(x1)
        
        x2 = self.branch2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1_adc(x2)
        x2 = self.fc_layers(x2)
        
        combined = torch.cat((x1, x2), dim=1)
        out = self.fc1_comb(combined)
        
        return out

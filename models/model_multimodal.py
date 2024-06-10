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

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        self.shared_conv1 = conv_block(3, 16)
        self.shared_conv2 = conv_block(16, 32)

        self.branch1 = nn.Sequential(
            conv_block(32, 16, kernel_size=1, padding=0),
            conv_block(16, 48),
            nn.MaxPool2d(2, 2),
            conv_block(48, 16, kernel_size=1, padding=0),
            conv_block(16, 64),
            conv_block(64, 16, kernel_size=1, padding=0),
            conv_block(16, 80),
            nn.MaxPool2d(2, 2),
        )

        self.branch2 = nn.Sequential(
            conv_block(32, 16, kernel_size=1, padding=0),
            conv_block(16, 48),
            nn.MaxPool2d(2, 2),
            conv_block(48, 16, kernel_size=1, padding=0),
            conv_block(16, 64),
        )

        self.fc1_t2 = nn.Linear(16 * 16 * 80, 500)
        self.fc1_adc = nn.Linear(22 * 22 * 64, 500)

        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(50, 2),
        )

        self.fc1_comb = nn.Linear(4, 2)

    def forward_branch(self, x, fc1_layer):
        x = self.shared_conv1(x)
        x = self.shared_conv2(x)
        x = self.relu(x)
        x = self.branch1(x) if fc1_layer == self.fc1_t2 else self.branch2(x)
        x = x.view(x.size(0), -1)
        x = fc1_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_layers(x)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_branch(x1, self.fc1_t2)
        x2 = self.forward_branch(x2, self.fc1_adc)
        combined = torch.cat((x1, x2), dim=1)
        out = self.fc1_comb(combined)
        return out

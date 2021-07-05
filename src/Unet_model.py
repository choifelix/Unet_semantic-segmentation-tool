# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:24:15 2021

@author: felix choi
"""
import torch
from torch import nn

from torchvision import datasets
from torchvision import transforms 

import torch.nn.functional as F

import matplotlib.pyplot as plt



## 2 - Load and Split the Data

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#U-net model for background segmentation

torch.cuda.empty_cache()
class ConvBlock(nn.Module):
    """
    Convolutional downsampling block
    
   
    """
    def __init__(self, inputs_ch, n_filters, dropout_prob=0, max_pooling=True):
        """
        Arguments:
            inputs             - previous block output
            n_filters         - Number of filters for the convolutional layers
            dropout_prob     - Dropout probability
            max_pooling     - Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_pooling  = max_pooling

        self.conv1      = nn.Conv2d(inputs_ch, n_filters, 3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(n_filters)
        self.relu       = nn.ReLU()
        self.conv2      = nn.Conv2d(n_filters, n_filters, 3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(n_filters)
        self.dropout    = nn.Dropout(dropout_prob)
        self.maxpool2d  = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        if self.dropout_prob > 0:
            x = self.dropout(x)

        if self.max_pooling :
            next_layer = self.maxpool2d(x)
        else :
            next_layer = x

        return next_layer, x


class Encoder(nn.Module):
    def __init__(self, input_size, stage_sizes=[64,128,256,512,1024], dropout_prob=0):
        super().__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(input_size, stage_sizes[0], dropout_prob=dropout_prob)] 
                                         + [ConvBlock(stage_sizes[i], stage_sizes[i+1], dropout_prob=dropout_prob) for i in range(len(stage_sizes)-2)]
                                         + [ConvBlock(stage_sizes[-2], stage_sizes[-1], dropout_prob=dropout_prob, max_pooling=False)])

    def forward(self, x):
        connections = []
        for block in self.conv_blocks:
            x, encoded = block(x)
            connections.append(encoded)

        return x, connections


class UpSamplingBlock(nn.Module):
    """
    Convolutional upsampling block


    """
    def __init__(self, expansive_input, contractive_input, n_filters):
        """
        Arguments:
            inputs             - previous block output
            n_filters         - Number of filters for the convolutional layers
            dropout_prob     - Dropout probability
            max_pooling     - Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """
        super().__init__()

        self.transconv  = nn.ConvTranspose2d(expansive_input, n_filters, 3, padding=1, stride=2, output_padding=1)
        self.relu       = nn.ReLU()
        # self.cat         = nn.cat()
        self.conv1      = nn.Conv2d(n_filters + contractive_input, n_filters, 3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(n_filters)
        self.conv2      = nn.Conv2d(n_filters, n_filters, 3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(n_filters)

        
    def forward(self, expansive, contractive):

        x = self.transconv(expansive)
        
        #handle size mismatch (non 2-powered size images)
        _, _, x_h, x_w = x.size()
        _, _, c_h, c_w = contractive.size()

        if x_h != c_h :
            x = nn.ZeroPad2d((0,0,1,0))(x)
            
        if x_w != c_w :
            x = nn.ZeroPad2d((1,0,0,0))(x)
        

        x = torch.cat((x, contractive), 1)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.relu(x)


        return x



class Decoder(nn.Module):
    def __init__(self, stage_sizes=[1024, 512, 256, 128, 64]):
         super().__init__()
         self.conv_blocks = nn.ModuleList([UpSamplingBlock(stage_sizes[i], stage_sizes[i+1], stage_sizes[i+1]) for i in range(len(stage_sizes)-1)] )

    def forward(self, x, connections):

        connections = connections[::-1]
        
        for i, block in enumerate(self.conv_blocks):

            x = block(x, connections[i])
        return x




class Unet_model(nn.Module):
    """
    Unet model using convBlcok and upSamplingBlock
    """
    def __init__(self, input_size, stage_sizes, head_filter, nb_classes, dropout_prob=0):
        """
        Arguments:
            input_size      - input size
            stage_sizes  - size for each block 
            output_sizes - output size
            nb_classes      - number of segmentation classes
        """
        super().__init__()
        
        self.encoder = Encoder(input_size , stage_sizes, dropout_prob)
        self.decoder = Decoder(stage_sizes[::-1])

        self.conv1   = nn.Conv2d(stage_sizes[0], head_filter, 3, padding='same')
        self.head    = nn.Conv2d(head_filter, nb_classes , 1, padding='same')


    def forward(self, x):
        encoded, connections = self.encoder(x)
        connections.pop(-1)

        out = self.decoder(encoded, connections)
        

        out = self.conv1(out)
        out = self.head(out)
        out = F.softmax(out, dim =1)

        return out
    









    
    
    










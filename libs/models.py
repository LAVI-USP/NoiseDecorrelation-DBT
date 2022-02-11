#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:34:44 2022

@author: Rodrigo
"""

from torch import nn

  
class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """
    def __init__(self,  num_filters = 64, inputLayer=False):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResidualBlock, self).__init__()
        
        
        in_filters = num_filters
        if inputLayer:
            in_filters = 1

        self.conv1 = nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class ResNetModified(nn.Module):
    def __init__(self, num_filters=64):
        super(ResNetModified, self).__init__()
        
        self.conv_first = nn.Conv2d(1, num_filters, 3, 1, 1)
        
        self.block1 = ResidualBlock(num_filters)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_last = nn.Conv2d(num_filters, 1, 3, 1, 1)
        
    def forward(self, x):
        
        identity = x
        out1 = self.conv_first(x)
        
        out2 = self.block1(out1)
        
        out3 = self.block2(out2)
        
        out3 = self.relu(out3 + out1)
        
        out4 = self.block3(out3)
        
        out5 = self.block4(out4)
        
        out5 = self.relu(out5 + out3)
        
        out = self.conv_last(out5)
        
        out = self.relu(identity + out)
        
        return out

class Gen(nn.Module):
    """
    Generator from WGAN
    """
    def __init__(self, num_filters=64):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(Gen, self).__init__()
        
        self.generator = ResNetModified(num_filters)

    def forward(self, x):

        return self.generator(x)

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator - 
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
    
class Disc(nn.Module):
    """
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    """
    def __init__(self, hidden_channels=64):
        super(Disc, self).__init__()
        self.upfeature = FeatureMapBlock(1, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn



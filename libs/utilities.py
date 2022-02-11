#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:33:54 2022

@author: Rodrigo
"""

import os 
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict

def load_model_gan(models, optimizers=None, schedulers=None, path_final_model='', path_pretrained_model=''):
    """Load pre-trained model, resume training or initialize from scratch."""
    
    epoch = 0
    
    model_names = ['gen_AB_state_dict', 'gen_BA_state_dict', 'dis_B_state_dict', 'dis_A_state_dict']
    opt_names = ['gen_optimizer_state_dict', 'dis_B_optimizer_state_dict', 'dis_A_optimizer_state_dict']
    sch_names = ['gen_scheduler_state_dict','dis_B_scheduler_state_dict','dis_A_scheduler_state_dict']
       
    # Resume training
    if os.path.isfile(path_final_model):
          
      checkpoint = torch.load(path_final_model)
      
      for model, model_name in zip(models, model_names):
          if model != None:
              model.load_state_dict(checkpoint[model_name])
      if optimizers != None:
          for optimizer, opt_name in zip(optimizers, opt_names):
              optimizer.load_state_dict(checkpoint[opt_name])
      if schedulers != None:
          for scheduler, sch_name in zip(schedulers, sch_names):
              scheduler.load_state_dict(checkpoint[sch_name])
      epoch = checkpoint['epoch'] + 1
      
      print('Loading model {} from epoch {}.'.format(path_final_model, epoch-1))
      
    elif os.path.isfile(path_pretrained_model):

      # Load a pre trained network 
      checkpoint = torch.load(path_pretrained_model)
      new_checkpoint = OrderedDict()
      for k, v in checkpoint['model_state_dict'].items():
          name = "generator." + k
          new_checkpoint[name] = v
          
      for idX, model in enumerate(models):
          model.load_state_dict(new_checkpoint)
          if idX == 1:
              break
      
      print('Initializing from scratch \nLoading pre-trained {}.'.format(path_pretrained_model))
               
    else:
      print('I couldnt find any model, I am just initializing from scratch.')
      
    return epoch


def image_grid_gan(real_A, fake_A, real_B, fake_B):
    """Return a 1x4 grid of the images as a matplotlib figure."""
    
    # Get from GPU
    real_A = real_A.to('cpu')
    real_B = real_B.to('cpu')
    fake_A = fake_A.to('cpu').detach()
    fake_B = fake_B.to('cpu').detach()
    
    # Create a figure to contain the plot.
    figure = plt.figure()
    
    plt.subplot(1,4,1)
    plt.imshow(torch.squeeze(real_A),'gray')
    plt.title("Correlated-real"); plt.grid(False)
    
    plt.subplot(1,4,2)
    plt.imshow(torch.squeeze(fake_B),'gray')
    plt.title("Uncorrelated-fake"); plt.grid(False)
    
    plt.subplot(1,4,3)
    plt.imshow(torch.squeeze(real_B),'gray')
    plt.title("Uncorrelated-real"); plt.grid(False)
    
    plt.subplot(1,4,4)
    plt.imshow(torch.squeeze(fake_A),'gray')
    plt.title("Correlated-fake"); plt.grid(False)
      
    return figure

def makedir(path2create):
    """Create directory if it does not exists."""
 
    error = 1
    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0
    
    return error

def set_requires_grad(nets, requires_grad = False):
    """Set the requires grad for networks."""
    
    if not isinstance(nets, list):
        nets = [nets]
    
    for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad


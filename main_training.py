#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:17:05 2022

@author: Rodrigo
"""

import matplotlib.pyplot as plt
import torch
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own codes
from libs.models import Gen, Disc
from libs.utilities import load_model_gan, image_grid_gan, makedir, set_requires_grad
from libs.dataset import VCTDataset

#%%

def train(gen_AB,
          gen_BA,
          dis_B,
          dis_A,
          gen_opt,
          dis_B_opt,
          dis_A_opt,
          epoch,
          train_loader,
          device,
          summarywriter,
          lmb_identity=0.1,
          lmb_cycle=10):

    # Enable trainning
    gen_AB.train()
    gen_BA.train()
    dis_B.train()
    dis_A.train()
    
    adv_criterion = torch.nn.MSELoss() 
    cycle_criterion = torch.nn.L1Loss()
    identity_criterion = torch.nn.L1Loss()

    for step, (real_A, real_B) in enumerate(tqdm(train_loader)):

        real_A = real_A.to(device)  # correlated noise
        real_B = real_B.to(device)  # uncorrelated noise
                
        # ---------------------
        
        ### Forward ###
        fake_B = gen_AB(real_A)     # G_A(A)
        rec_A  = gen_BA(fake_B)     # G_B(G_A(A))
        fake_A = gen_BA(real_B)     # G_B(B)
        rec_B  = gen_AB(fake_A)     # G_A(G_B(B))
        
        idt_B = gen_AB(real_B)     # G_A(B)
        idt_A = gen_BA(real_A)     # G_B(A)
        
        ######### Update Generator #########
        
        set_requires_grad([dis_A, dis_B], requires_grad = False)
        
        # Zero all grads            
        gen_opt.zero_grad()


        # Identity Loss
        identity_loss_A = identity_criterion(idt_B, real_B)
        identity_loss_B = identity_criterion(idt_A, real_A)
        
        gen_identity_loss = (identity_loss_A + identity_loss_B) * lmb_identity
        
        # Adversarial Loss
        disc_fake_A_hat = dis_A(fake_A)
        adv_loss_BA = adv_criterion(disc_fake_A_hat, torch.ones_like(disc_fake_A_hat)) 
        
        disc_fake_B_hat = dis_B(fake_B)
        adv_loss_AB = adv_criterion(disc_fake_B_hat, torch.ones_like(disc_fake_B_hat)) 
        
        gen_adversarial_loss = (adv_loss_AB + adv_loss_BA) * 0.5
        
        # Cycle-consistency Loss
        cycle_loss_AA = cycle_criterion(rec_A, real_A)
        cycle_loss_BB = cycle_criterion(rec_B, real_B)
        
        gen_cycle_loss = (cycle_loss_AA + cycle_loss_BB) * lmb_cycle
        
        # Total gen loss
        gen_loss = gen_identity_loss + gen_adversarial_loss + gen_cycle_loss
        
        ### Backpropagation ###
        # Calculate all grads
        gen_loss.backward()
        
        # Update weights and biases based on the calc grads 
        gen_opt.step()
        
        
        ######### Update Discriminator A #########
        set_requires_grad(dis_A, requires_grad = True)
        
        # Zero all grads 
        dis_A_opt.zero_grad()
        
        disc_fake_A_hat = dis_A(fake_A.detach()) # Detach generator
        disc_fake_A_loss = adv_criterion(disc_fake_A_hat, torch.zeros_like(disc_fake_A_hat))
        
        disc_real_A_hat = dis_A(real_A)
        disc_real_A_loss = adv_criterion(disc_real_A_hat, torch.ones_like(disc_real_A_hat))
        disc_loss_A = (disc_fake_A_loss + disc_real_A_loss) * 0.5 
        
        ### Backpropagation ###
        # Calculate all grads
        disc_loss_A.backward()
        
        # Update weights and biases based on the calc grads 
        dis_A_opt.step()
        
        ######### Update Discriminator B #########
        set_requires_grad(dis_B, requires_grad = True)
        
        # Zero all grads 
        dis_B_opt.zero_grad()
        
        disc_fake_B_hat = dis_B(fake_B.detach()) # Detach generator
        disc_fake_B_loss = adv_criterion(disc_fake_B_hat, torch.zeros_like(disc_fake_B_hat))
        
        disc_real_B_hat = dis_B(real_B)
        disc_real_B_loss = adv_criterion(disc_real_B_hat, torch.ones_like(disc_real_B_hat))
        disc_loss_B = (disc_fake_B_loss + disc_real_B_loss) * 0.5 
        
        ### Backpropagation ###
        # Calculate all grads
        disc_loss_B.backward()
        
        # Update weights and biases based on the calc grads 
        dis_B_opt.step()
        
        # ---------------------

        # Print images to tensorboard
        if step % 20 == 0:
            # Write Gen Loss to tensorboard
            summarywriter.add_scalar('Loss/GEN-Iden',
                                     gen_identity_loss.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_scalar('Loss/GEN-Adv',
                                     gen_adversarial_loss.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_scalar('Loss/GEN-Cyc',
                                     gen_cycle_loss.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_scalar('Loss/GEN-Total',
                                     gen_loss.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_scalar('Loss/DIS_A',
                                     disc_loss_A.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_scalar('Loss/DIS_B',
                                     disc_loss_B.item(),
                                     epoch * len(train_loader) + step)

            summarywriter.add_figure('Plot/train', 
                                     image_grid_gan(real_A[18,0,5:-5,5:-5], 
                                                    fake_A[18,0,5:-5,5:-5], 
                                                    real_B[18,0,5:-5,5:-5], 
                                                    fake_B[18,0,5:-5,5:-5]),
                                     epoch * len(train_loader) + step,
                                     close=True)

#%%

if __name__ == '__main__':
        
    # Noise scale factor
    mAsFullDose = 360
    mAsLowDose = 90
    
    red_factor = mAsLowDose / mAsFullDose
    
    path_data = "data/"
    path_models = "final_models/"
    path_logs = "final_logs/{}-{}mAs".format(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()), mAsLowDose)
    
    path_final_model = path_models + "cGAN_ResNet_Decor-{}mAs.pth".format(mAsLowDose)
    
    LRg = 1e-4/10
    LRd = 1e-4/10
    batch_size = 32
    n_epochs = 30
    
    dataset_path = '{}DBT-PMMA_Decor_training_{}mAs.h5'.format(path_data,mAsLowDose)
    
    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)
    
    makedir(path_models)
    makedir(path_logs)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    gen_AB = Gen()
    gen_BA = Gen()
    dis_B = Disc()
    dis_A = Disc()
        
    # Create the optimizer and the LR scheduler
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=LRg, betas=(0.5, 0.999))
    dis_B_opt = torch.optim.Adam(dis_B.parameters(), lr=LRd, betas=(0.5, 0.999))
    dis_A_opt = torch.optim.Adam(dis_A.parameters(), lr=LRd, betas=(0.5, 0.999))
    
    gen_sch = torch.optim.lr_scheduler.MultiStepLR(gen_opt, milestones=[10, 20], gamma=0.5)
    dis_B_sch = torch.optim.lr_scheduler.MultiStepLR(dis_B_opt, milestones=[10, 20], gamma=0.5)
    dis_A_sch = torch.optim.lr_scheduler.MultiStepLR(dis_A_opt, milestones=[10, 20], gamma=0.5)
    
    
    # Send it to device (GPU if exist)
    gen_AB = gen_AB.to(device)
    gen_BA = gen_BA.to(device)
    dis_B = dis_B.to(device)
    dis_A = dis_A.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    start_epoch = load_model_gan([gen_AB,gen_BA,dis_B,dis_A], 
                            [gen_opt, dis_B_opt, dis_A_opt], 
                            [gen_sch, dis_B_sch, dis_A_sch],
                            path_final_model=path_final_model,
                            path_pretrained_model=None)

    start_epoch = 0
    
    # Create dataset helper
    train_set = VCTDataset(dataset_path, red_factor, vmin=17276., vmax=10084.)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              pin_memory=True)
            
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs):
        
      print("Epoch:[{}] LR:{}".format(epoch, gen_opt.state_dict()['param_groups'][0]['lr']))
    
      # Train the model for 1 epoch
      train(gen_AB, gen_BA, dis_B, dis_A, gen_opt, dis_B_opt, dis_A_opt, epoch, train_loader, device, summarywriter) 
    
      # Update LR
      gen_sch.step()
      dis_A_sch.step()
      dis_B_sch.step()
    
      # Save the model
      torch.save({
               'epoch': epoch,
               'gen_AB_state_dict': gen_AB.state_dict(),
               'gen_BA_state_dict': gen_BA.state_dict(),
               'dis_A_state_dict': dis_A.state_dict(),
               'dis_B_state_dict': dis_B.state_dict(),
               'gen_optimizer_state_dict': gen_opt.state_dict(),
               'dis_A_optimizer_state_dict': dis_A_opt.state_dict(),
               'dis_B_optimizer_state_dict': dis_B_opt.state_dict(),
               'gen_scheduler_state_dict': gen_sch.state_dict(),
               'dis_A_scheduler_state_dict': dis_A_sch.state_dict(),
               'dis_B_scheduler_state_dict': dis_B_sch.state_dict(),
               }, path_final_model)
      

# Required libraries -----------------------------------------------------------------------------------------------------------
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
import time

import vitaldb
from model import Generator
from model import Discriminator
from utils import findMinMax
from utils import minMax
from utils import weights_init_normal
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import Adam
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

# Set CUDA -------------------------------------------------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'


# Model parameters -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='data1.npz', help='datasets location')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--workers', type=int, default=2, help='the number of workers')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--beta1', type=int, default=1, help='betal value for Adam optimizer')
parser.add_argument('--seed', type=int, default=30, help='random seed')
parser.add_argument('--model_info', type=str, default='/models_info.pth', help='save models\' information')
parser.add_argument('--ppg_test', type=str, default='/ppg_test.pth', help='save ppg test datasets')
parser.add_argument('--abp_test', type=str, default='/abp_test.pth', help='save abp test datasets')
opt = parser.parse_args()

## Set the computation device -----------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Initialize models -------------------------------------------------------------------------------------------------
## Initalize Generator and Discriminator
netG_P2A = Generator(opt.input_nc, opt.output_nc).to(device)
netG_A2P = Generator(opt.output_nc, opt.input_nc).to(device)
netD_PPG = Discriminator(opt.input_nc).to(device)
netD_ABP = Discriminator(opt.input_nc).to(device)

## Weight Initialization
netG_P2A.apply(weights_init_normal)
netG_A2P.apply(weights_init_normal)
netD_PPG.apply(weights_init_normal)
netD_ABP.apply(weights_init_normal)

# Hyperparameter -----------------------------------------------------------------------------------------------------------
## Optimizers
optimizer_G = optim.Adam(opt.itertools.chain(netG_P2A.parameters(), netG_A2P.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_PPG = optim.Adam(netD_ABP.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_ABP = optim.Adam(netD_PPG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

## Cross-function
criterion_gan = nn.MSELoss().to(device)      # How the discriminator discriminates the data whether it is generated or not
criterion_identity = nn.L1Loss().to(device)  # How different the generated data is from the original data through forward OR backward transformations
criterion_cycle = nn.L1Loss().to(device)     # How different the generated data is from the original data throu

## learning rate schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 0.95 ** epoch)
lr_scheduler_D_PPG = torch.optim.lr_scheduler.LambdaLR(optimizer_D_PPG, lr_lambda=lambda epoch: 0.95 ** epoch)
lr_scheduler_D_ABP = torch.optim.lr_scheduler.LambdaLR(optimizer_D_ABP, lr_lambda=lambda epoch: 0.95 ** epoch)

# Initaliza loss value
min_loss_G = float('inf')

# DataLoader      -----------------------------------------------------------------------------------------------------------
## Load cache file 
ppg_abp_sets = np.load(opt.datapath)
ppg_sets = ppg_abp_sets['ppg_sets']
abp_sets = ppg_abp_sets['abp_sets']
ppg_abp_sets.close()                    # For memory efficiency, close the file

## Split data into training and testing datasets
ppg_train, ppg_test, abp_train, abp_test = train_test_split(ppg_sets, abp_sets, test_size=0.2, random_state=opt.seed)
torch.save(ppg_test, opt.ppg_test)
torch.save(abp_test, opt.abp_test)

## Normalization
norm_ppg_train, norm_abp_train = minMax(ppg_train), minMax(abp_train)

## Change data type
norm_ppg_train = torch.tensor(norm_ppg_train, dtype=torch.float32)
norm_abp_train = torch.tensor(norm_abp_train, dtype=torch.float32)

## Make the datasets as a TensorDataset
ds_train = TensorDataset(norm_ppg_train, norm_abp_train)

## Create the dataloader
loader_train = DataLoader(ds_train, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)


# Train -----------------------------------------------------------------------------------------------------------
for epoch in range(opt.epoch, opt.n_epochs):
    print("%d epoch-------------------------------"%(epoch+1))
    for real_ppg, real_abp in tqdm(loader_train):
        time.sleep(0.01)
        #########################################
        # Generator PPG2ABP and ABP2PPG
        # 1. GAN loss
        # 2. Cycle loss
        # 3. Identity loss
        # 4. Total loss
        #########################################
        
        # Forward------------------------------------------
        # Transform the datasets
        real_ppg_3d = real_ppg.unsqueeze(0).transpose(0,1).to(device)
        real_abp_3d = real_abp.unsqueeze(0).transpose(0,1).to(device)
        
        # 1. Generate fake data
        fake_abp = netG_P2A(real_ppg_3d)
        fake_ppg = netG_A2P(real_abp_3d)

        recovered_ppg = netG_A2P(fake_abp)
        recovered_abp = netG_P2A(fake_ppg)
        
        identity_abp = netG_P2A(real_abp_3d)
        identity_ppg = netG_A2P(real_ppg_3d)
        
        
        # Backward generator path --------------------------
        discFakeppg = netD_PPG(fake_ppg)
        discFakeabp = netD_ABP(fake_abp)
        discCycleppg = netD_PPG(recovered_ppg)
        discCycleabp = netD_ABP(recovered_abp)
        
        
        # 1. GAN loss
        loss_ppg = criterion_gan(discFakeppg, torch.ones_like(discFakeppg))
        loss_abp = criterion_gan(discFakeabp, torch.ones_like(discFakeabp))
        
        
        # 2. Cycle loss
        loss_cycle_abp = criterion_cycle(recovered_abp, real_abp_3d)
        loss_cycle_ppg = criterion_cycle(recovered_ppg, real_ppg_3d)
        
        # 3. Identity loss
        loss_identity_ppg = criterion_identity(identity_ppg, real_ppg_3d)
        loss_identity_abp = criterion_identity(identity_abp, real_abp_3d)
        
        
        # 4. Total loss = GAN loss + Cycle loss + Identity loss
        loss_G = (loss_ppg + loss_abp) + \
                 (loss_identity_ppg + loss_identity_abp) * 5.0 + \
                 (loss_cycle_ppg + loss_cycle_abp) * 10.0
        
        optimizer_G.zero_grad()
        loss_G.backward()   # Calculates the loss of the loss function
        optimizer_G.step()
        
        
        # Backward generator path --------------------------
        
        discFakeppg = netD_PPG(fake_ppg.detach())
        discRealppg = netD_PPG(real_ppg_3d)
        
        loss_D_Real_ppg = criterion_gan(discFakeppg, torch.ones_like(discFakeppg))
        loss_D_Fake_ppg = criterion_gan(discRealppg, torch.zeros_like(discRealppg))
                        
        loss_D_PPG = (loss_D_Real_ppg + loss_D_Fake_ppg) * 0.5
        optimizer_D_PPG.zero_grad()
        loss_D_PPG.requires_grad_(True)
        loss_D_PPG.backward()
        optimizer_D_PPG.step()
        
        discFakeabp = netD_ABP(fake_abp.detach())
        discRealabp = netD_ABP(real_abp_3d)
        
        loss_D_Real_abp = criterion_gan(discFakeabp, torch.ones_like(discFakeabp))
        loss_D_Fake_abp = criterion_gan(discRealabp, torch.zeros_like(discRealabp))
        
        loss_D_ABP = (loss_D_Real_abp + loss_D_Fake_abp) * 0.5
        optimizer_D_ABP.zero_grad()
        loss_D_ABP.requires_grad_(True)
        loss_D_ABP.backward()
        optimizer_D_ABP.step()
      
    # Print loss
    print("Epoch [{}/{}], loss_G: {:.4f}, loss_D_ABP: {:.4f}, loss_D_PPG: {:.4f}".format(
           epoch+1, opt.n_epochs, loss_G, loss_D_ABP, loss_D_PPG))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_PPG.step()
    lr_scheduler_D_ABP.step()

    # Save models checkpoints
    if min_loss_G >= loss_G:
      min_loss_G = loss_G
      torch.save({
            'epoch_info': epoch,
            'netG_P2A_state_dict': netG_P2A.state_dict(),
            'netG_A2P_state_dict': netG_A2P.state_dict(), 
            'netD_ABP_state_dict': netD_ABP.state_dict(), 
            'netD_PPG_state_dict': netD_PPG.state_dict(), 
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_PPG_state_dict': optimizer_D_PPG.state_dict()
            'optimizer_D_ABP_state_dict': optimizer_D_ABP.state_dict(),
            'criterion_gan_state_dict': criterion_gan.state_dict(),
            'criterion_identity_state_dict': criterion_identity.state_dict(),
            'criterion_cycle_state_dict': criterion_cycle.state_dict(),
            'lr_scheduler_G_state_dict': lr_scheduler_G.state_dict(),
            'lr_scheduler_D_PPG_state_dict': lr_scheduler_D_PPG.state_dict(),
            'lr_scheduler_D_ABP_state_dict': lr_scheduler_D_ABP.state_dict()
      }, opt.model_info)
      
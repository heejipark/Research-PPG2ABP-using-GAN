###### Test ######
from utils import findMinMax
from utils import minMax
import argparse
import torch
from model import Generator
from model import Discriminator
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from visualization import graph

# Model parameters -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='data1.npz', help='datasets location')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--model_info', type=str, default='/models_info.pth', help='checkpoint for models')
parser.add_argument('--test_dataset', type=str, default='/loader_test.pth', help='save test datasets')
parser.add_argument('--ppg_test', type=str, default='/ppg_test.pth', help='save ppg test datasets')
parser.add_argument('--abp_test', type=str, default='/abp_test.pth', help='save abp test datasets')
parser.add_argument('--output', type=str, default='/output/.pth', help='save abp test datasets')
opt = parser.parse_args()

## Load test dataset
ppg_test = torch.load(opt.ppg_test)
abp_test = torch.load(opt.abp_test)

## Normalization
norm_ppg_test, norm_abp_test =  minMax(ppg_test), minMax(abp_test)

## Change data type
norm_ppg_test = torch.tensor(norm_ppg_test, dtype=torch.float32)
norm_abp_test = torch.tensor(norm_abp_test, dtype=torch.float32)

## Make the datasets as a TensorDataset
ds_test = TensorDataset(norm_ppg_test, norm_abp_test)

## Create the dataloader
loader_test = DataLoader(ds_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

# Set the computation device -----------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Definition of models -------------------------------------------------------------------------------------------------
## Generator and Discriminator
netG_P2A = Generator(opt.input_nc, opt.output_nc).to(device)
netG_A2P = Generator(opt.output_nc, opt.input_nc).to(device)
netD_PPG = Discriminator(opt.input_nc).to(device)
netD_ABP = Discriminator(opt.input_nc).to(device)

# Load state dicts
checkpoint = torch.load(opt.model_info)
netG_P2A.load_state_dict(checkpoint['netG_P2A_state_dict'])
netG_A2P.load_state_dict(checkpoint['netG_A2P_state_dict'])
epoch = checkpoint['epoch_info']

# Set model's test mode
netG_P2A.eval()
netG_A2P.eval()


# Create output dirs if they don't exist
dataNum = 1
for real_ppg, real_abp in loader_test:
    # Set model input
    real_ppg_3d = real_ppg.unsqueeze(0).transpose(0,1).to(device)
    real_abp_3d = real_abp.unsqueeze(0).transpose(0,1).to(device)

    # Generate output
    fake_abp = netG_P2A(real_ppg_3d).data
    fake_ppg = netG_A2P(real_abp_3d).data
    
    # Find min-max value from test datasets
    max_ppg, min_ppg = findMinMax(ppg_test)[0], findMinMax(ppg_test)[1]
    max_abp, min_abp = findMinMax(abp_test)[0], findMinMax(abp_test)[1]

    # Revert the data from normalized one
    revert_real_abp = real_abp_3d * (max_abp - min_abp) + min_abp
    revert_real_ppg = real_ppg_3d * (max_ppg - min_ppg) + min_ppg
    revert_fake_abp = fake_abp * (max_abp - min_abp) + min_abp
    revert_fake_ppg = fake_ppg * (max_ppg - min_ppg) + min_ppg

    # Save data file 
    resultfile = opt.output + 'data%d.npz'%(dataNum)
    np.savez(resultfile, real_abp=revert_real_abp.squeeze().cpu(), fake_abp=revert_fake_abp.squeeze().cpu(), 
             real_ppg=revert_real_ppg.squeeze().cpu(), fake_ppg=revert_fake_ppg.squeeze().cpu()) # Save cahce file
    dataNum += 1


# Make a graph with index 0 dataset
graph(0)
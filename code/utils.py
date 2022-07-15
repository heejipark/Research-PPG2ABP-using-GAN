import numpy as np
import torch

# Find minimum and maximum values
def findMinMax(datasets):
  return [np.max(datasets), np.min(datasets)]

# Min-Max Normalization
def minMax(datasets):
  maxVal, minVal = findMinMax(datasets)[0], findMinMax(datasets)[1]
  normVal = (datasets - minVal) / (maxVal - minVal)
  return normVal

# weights initalization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

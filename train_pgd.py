# Source: https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/tree/master
# Filename: train_pgd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from models import * # ResNet18, WideResNet

from util import *

### HYPERPARAMETER ###
learning_rate = 0.1
# PGD parameters
train_pgd_iter = 10 # num iterations
test_pgd_iter = 20 # num iterations
epsilon = 0.0314 # maximum distortion = 8/255
alpha = 0.00784 # attack step size = 2/255
rand_init = True
save_file_name = 'resnet18'
######################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = RobustModel(device)
adversary = Adversary(model.model)

model.train(0, adversary) # trains for one epoch
# test(19, save_path)
    
# Train
# for epoch in range(0, 90):
#     adjust_learning_rate(optimizer, epoch)
#     train(epoch)
#     if (epoch+1) % 10 == 0:
#         test(epoch, save_path)
    
# test_autoattack()
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

train_start_epoch = 0
load_model = False
saved_file = "/saved_checkpoint+_89.pt"
######################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment = RobustExperiment(device)
if load_model:
    train_start_epoch = experiment.load_model(saved_file)
adversary = Adversary(experiment, device)


# Train
for epoch in range(train_start_epoch, 90):
    experiment.adjust_learning_rate(epoch)
    experiment.train(epoch, adversary)

    if (epoch+1) % 5 == 0:
        experiment.test(epoch, adversary)
    if (epoch+1) % 10 == 0:
        adversary.test_autoattack(epoch=epoch, full_test=False, num_tests=100)

adversary.test_autoattack(full_test=True)
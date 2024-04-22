
import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from models import *

# pip install git+https://github.com/fra31/auto-attack
from autoattack import AutoAttack

class RobustExperiment():
    def __init__(
            self,
            device,
            lr=0.1,
            train_pgd_iter=10,
            test_pgd_iter=20,
            save_name="resnet18",
            testing=False):
        self.device = device

        # self.model = WideResNet(num_classes=10, depth=34, widen_factor=10, activation='ReLU')
        self.model = ResNet18()
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        cudnn.benchmark = True

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

        self.train_loader, self.test_loader = self._load_all_dataset()

        # PGD attack iteration
        self.train_pgd_iter = train_pgd_iter
        self.test_pgd_iter = test_pgd_iter

        self.checkpoint_dir = "checkpoint"
        self.save_name = save_name
        self.save_dir = os.path.join(self.checkpoint_dir, self.save_name)
        # Create all save directories and files
        if not testing:
            if not os.path.isdir(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            if os.path.isdir(self.save_dir):
                self.save_dir += "_new"
            else:
                os.mkdir(self.save_dir)
            self._init_train_log_file()
            self._init_autoattack_log_file()
            self._init_test_log_file()


    def _load_all_dataset(self):
        """
        Returns train DataLoader and test DataLoader
        """
        train_loader = self._load_train_dataset()
        test_loader = self._load_test_dataset()
        return train_loader, test_loader

    def _load_train_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        return train_loader

    def _load_test_dataset(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
        return test_loader
    
    def _init_train_log_file(self):
        # Create and remove contents of the file
        self.train_log = self.save_dir + "/train_log.txt"
        open(self.train_log, 'w').close()

    def _init_autoattack_log_file(self):
        # Create and remove contents of the file
        self.autoattack_log = self.save_dir + "/autoattack_log.txt"
        open(self.autoattack_log, 'w').close()

    def _init_test_log_file(self):
        self.pgd_test_log = self.save_dir + "/pgd_test_log.txt"
        open(self.pgd_test_log, 'w').close()

    def load_model(self, saved_file):
        checkpoint = torch.load(self.save_dir + saved_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def train(self, epoch, adversary):
        print('\n[ Train epoch: %d ]' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            adv = adversary.perturb(inputs, targets, self.train_pgd_iter)
            adv_outputs = self.model(adv)
            loss = self.criterion(adv_outputs, targets)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # if batch_idx % 10 == 0:
            #     print('\nCurrent batch:', str(batch_idx))
            #     print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            #     print('Current adversarial train loss:', loss.item())

        print('\nTotal train robust accuarcy:', 100. * correct / total)
        print('Total train robust loss:', train_loss)

    def test(self, epoch, adversary):
        print('\n[ Test epoch: %d ]' % epoch)
        self.model.eval()
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                total += targets.size(0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                benign_loss += loss.item()

                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()

                # if batch_idx % 10 == 0:
                #     print('\nCurrent batch:', str(batch_idx))
                #     print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                #     print('Current benign test loss:', loss.item())

                adv = adversary.perturb(inputs, targets, self.test_pgd_iter)
                adv_outputs = self.model(adv)
                loss = self.criterion(adv_outputs, targets)
                adv_loss += loss.item()

                _, predicted = adv_outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

                # if batch_idx % 10 == 0:
                #     print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                #     print('Current adversarial test loss:', loss.item())
        
        benign_acc_log = '\nTotal benign test accuracy: ' + str(100. * benign_correct / total)
        adv_acc_log = 'Total adversarial test accuracy: ' + str(100. * adv_correct / total)
        benign_loss_log = 'Total benign test loss: ' + str(benign_loss)
        adv_loss_log = 'Total adversarial test loss: ' + str(adv_loss)

        print(benign_acc_log)
        print(adv_acc_log)
        print(benign_loss_log)
        print(adv_loss_log)
        
        # Save log
        with open(self.pgd_test_log, 'a') as log:
            log.write("Epoch {}".format(epoch))
            log.write(benign_acc_log)
            log.write('\n' + adv_acc_log + '\n\n')
            log.flush()

        # Save model
        save_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': benign_loss,
            'adv_loss': adv_loss
        }
        torch.save(save_state, self.save_dir + "/saved_checkpoint_{}.pt".format(epoch))

    def adjust_learning_rate(self, epoch):
        lr = self.lr
        if epoch >= 30:
            lr /= 10
        if epoch >= 60:
            lr /= 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Adversary(object):
    def __init__(
            self,
            robust_exp,
            device,
            rand_init=True,
            epsilon=8/255,
            alpha=2/255,
            testing=False):
        self.device = device
        self.testing = testing

        self.exp = robust_exp
        self.model = robust_exp.model

        # PGD hyperparameters
        self.rand_init = rand_init # atk noise starts random
        self.epsilon = epsilon # maximum distortion = 8/255
        self.alpha = alpha # attack step size = 2/255

        # AutoAttack hyperparameters
        self.autoattack = AutoAttack(
            self.model,
            norm= 'Linf', # 'L2',
            eps=self.epsilon,
            verbose=True,
            device=self.device,
            log_path= None if testing else self.exp.autoattack_log)
        self.test_atks = ["apgd-ce", "apgd-dlr", "square", "fab-t"]
        self.num_total_test_imgs = 10000

    def perturb(self, x_natural, y, pgd_iter):
        x = x_natural.detach()
        # Random initialization
        if self.rand_init:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(pgd_iter):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(
                    torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon
                    )
            x = torch.clamp(x, 0, 1)
        return x
    
    def test_autoattack(self, epoch=None, full_test=False, num_tests=100):
        if not full_test and num_tests > self.num_total_test_imgs:
            print("Couldn't run AutoAttack test - # of tests > total # of test images")
            return
        
        self.model.eval()
        with torch.no_grad():
            x_test = [x for (x,y) in self.exp.test_loader]
            x_test = torch.cat(x_test, 0)

            y_test = [torch.Tensor(y) for (x,y) in self.exp.test_loader]
            y_test = torch.cat(y_test, 0)

            if not full_test:
                rand_ind = np.random.choice(
                    self.num_total_test_imgs, num_tests, replace=False)
                x_test = x_test[rand_ind].to(self.device)
                y_test = y_test[rand_ind].to(self.device)
            else:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

            if not self.testing:
                with open(self.exp.autoattack_log, 'a') as log:
                    log.write("AutoAttack test @ Epoch {}\n".format(epoch))
                    log.flush()

            self.autoattack.verbose = True
            self.autoattack.attacks_to_run = self.test_atks
            dict_adv = self.autoattack.run_standard_evaluation_individual(
                x_test, y_test, bs=100)
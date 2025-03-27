from __future__ import division
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import sys
import copy
import time
#import cv2
import os
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import NullFormatter
from sklearn import manifold
from torchvision.utils import save_image
from attacks import AttackerPolymer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__=='__main__':

    from models.resnet import ResNet18
    from robustbench import benchmark
    net=ResNet18().to(device)
    net.load_state_dict(torch.load('./model/cifar10_resnet18_tea_norm_2.pkl'))


    clean_acc,robust_acc=benchmark(net.eval(),model_name='CIFAR10',n_examples=10000,dataset='cifar10',threat_model='Linf',eps=8/255,device='cuda',to_disk=True)
    print(clean_acc,robust_acc)

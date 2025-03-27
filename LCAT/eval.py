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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from torchvision.utils import save_image

from models.densenet import DenseNet,DenseNet121,DenseNet201,DenseNet161,densenet_cifar
from models.efficientnet import EfficientNetB0
from vgg import VGG


from models.resnet import ResNet18
from models.wideresnet import wrn28x5,wrn28x10,wrn34x10
from autoattack import AutoAttack
from attacks import AttackerPolymer

from pylab import *
#from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
batch_size=128
mean,std=(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing data..')
mean,std=(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean,std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean,std)

])



testset = torchvision.datasets.CIFAR10(
    root='datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net= ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()





class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha,(self.step + 1)/(self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


def eval_cifar():
    #attck_list = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
    attack_list=['apgd-ce','square']
    net.eval()
    #net_orc.eval()
    correct=0
    tmp_len=0
    adversary.attacks_to_run=attack_list
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs,targets=inputs.to(device),targets.to(device)
        inputs_adv=adversary.run_standard_evaluation(inputs,targets)
        outs=net(inputs_adv)
        
        num_classes=10
        out_=outs.topk(num_classes)
        max_values = out_.values
        out_= torch.relu((max_values[:, 0] - max_values[:,num_classes-1]))
        pred=outs.max(1)[1]
        correct=pred.eq(targets).sum()
        print(correct/len(inputs))
        tmp_len+=len(inputs)

        correct=correct.item()
        print("Processing: {}_th batch Test acc is {:.3f} {}/{}|OUT_MIN {:.4f} ".format(batch_idx, correct / tmp_len * 100,correct,tmp_len,
                                                                                      (out_).mean().detach().cpu().numpy()))
        
    print('Acc is {:.4f}'.format(100*correct/len(test_loader.dataset)))#len(test_loader.dataset)
    return 0




    print(loss/epo)
def test(epoch):
    net.eval()
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            #inputs=inputs+ torch.randn_like(inputs, device='cuda') * noise_sd_gauss+ (torch.rand_like(inputs, device='cuda') - 0.5) * 2 * noise_sd_unif * np.sqrt(3)
            outputs=net(inputs)
            pred=outputs.max(1)[1]
            correct += pred.eq(targets).sum().item()
        correct=int(correct)
        tmp_acc = 100 * correct / len(test_loader.dataset)
        print('Test accuracy is {:.4f}'.format(tmp_acc))
    return tmp_acc


if __name__=='__main__':
    net=ResNet18(mu=1)
   
   
    net = net.to(device)

    net.load_state_dict(torch.load('./model/cifar10_resnet18_tea_lact.pkl'))
   
    
    test(0)

    adversary = AutoAttack(net, norm='Linf', eps=8/255, version='standard',verbose=True)
    
    eval_cifar()

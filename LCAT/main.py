'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import copy
import math
import random
import cv2


from models.densenet import DenseNet,DenseNet121,DenseNet201,DenseNet161,densenet_cifar
from models.efficientnet import EfficientNetB0

#from models.resnet_ori import ResNet18
from models.resnet import ResNet18
from models.wideresnet import wrn28x5,wrn28x10,wrn34x10

from attacks import AttackerPolymer
from autoattack import AutoAttack
import random


#from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import copy
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step')
parser.add_argument('--step-size', type=float, default=3/255, help='step size')
parser.add_argument('--resume', '-r',default=False, action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print("Device is {}".format(device))
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

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

trainset = torchvision.datasets.CIFAR10I(
    root='datasets/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10I(
    root='/datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#Pair [1,9] [3,5] [0,1,8,9][2,3,4,5,6,7]
#list=[0,1,8,9]

# Model
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
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
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





start_epoch=0
end_epoch=args.epochs
criterion = nn.CrossEntropyLoss()

kl=nn.KLDivLoss(reduction='batchmean')


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  
        pt = torch.exp(-ce_loss)  
        focal_loss = (1 - pt) ** self.gamma * ce_loss 
        return focal_loss


mse_cre=nn.MSELoss()
focal=FocalLoss()

def consist_loss_s(logits,logits_adv,target,mu=1,beta=2,num_classes=10):
    #alpha=1

    bs=len(logits_adv)
    #one_hots=F.one_hot(target,num_classes)
    max_adv=logits_adv.topk(num_classes)
    max_indexes_adv = max_adv[1]

    max_value_adv=(logits_adv*F.one_hot(max_indexes_adv[:,0],num_classes)).sum(dim=-1)
    min_value_adv =(logits_adv*F.one_hot(max_indexes_adv[:, num_classes-1],num_classes)).sum(dim=-1)

    #+max_value_inv = (logits_adv * F.one_hot(max_indexes[:, 1], num_classes)).sum(dim=-1)
    diff_adv=(max_value_adv - min_value_adv)
    loss_r=(torch.relu((beta-diff_adv)))
    loss_l=(torch.relu((diff_adv-beta)))

    max = logits.topk(num_classes)
    max_indexes = max[1]

    max_value= (logits* F.one_hot(max_indexes[:, 0], num_classes)).sum(dim=-1)
    min_value = (logits * F.one_hot(max_indexes[:, num_classes - 1], num_classes)).sum(dim=-1)
    # +max_value_inv = (logits_adv * F.one_hot(max_indexes[:, 1], num_classes)).sum(dim=-1)
    diff = (max_value - min_value)

    loss_diff=torch.norm(diff.mean()-diff_adv.mean(),2)


    # values=logits_adv[np.where((1-F.one_hot(max_indexes[:,0],num_classes).detach().cpu())==1)]
    # values=values.view(-1,num_classes-1)
    # values=F.softmax(values,dim=-1)
    # values_log=torch.log(values+1e-12)
    #entr=(-values_log*values).sum()
    loss_norm=(loss_l+loss_r+loss_diff)
    #loss_norm=torch.norm(max_value-min_value-2,0)



    loss_norm=loss_norm*(1-F.softmax(loss_norm,dim=-1))
    loss_norm=loss_norm.sum()

    logits_adv = F.softmax(logits_adv, dim=-1)
    # mask=(logits_adv>=alpha)
    # mask=(mask+0)*one_hots
    #
    # logits_adv=logits_adv*(1-mask)+mask
    logits_adv=torch.log(logits_adv+1e-12)


    rand_=np.random.rand(1)
    if rand_>9/10:
        print("Current loss_ is {:.5f} vs {:.5f}|{:.5f} {:.5} ".format(loss_norm.item(), F.nll_loss(logits_adv,target).item(),max_value.mean().item(), min_value.mean().item()))
    loss=focal(logits_adv,target)+mu*loss_norm/bs
    return loss



# Training
def train(epoch):
    global flag
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    start = 0

    for batch_idx, (inputs, targets,index) in enumerate(train_loader):
        

        end=start+len(inputs)

        inputs, targets = inputs.to(device), targets.to(device)
        # net.eval()
        # inputs_adv = Attackers.run_specified('PGD_5', net, inputs, targets, return_acc=False)
        #inputs_adv=adversary.run_standard_evaluation(inputs,targets)

        
        fea_inputs=[]
        def hook_fn_fea(module, inp, out):
            fea_inputs.append(out)
        fea2_inputs = []
        def hook_fn_fea2(module, inp, out):
            fea2_inputs.append(out)

        # h_ grad = net.linear.register_full_backward_hook(hook_fn_grad)
        # h_fea = net.layer1.register_forward_hook(hook_fn_fea)
        #


        net.train()
        optimizer.zero_grad()
        #outputs= net(inputs)
        # fea1_ori=copy.deepcopy(fea_inputs[0].detach())
        # fea2_ori = copy.deepcopy(fea2_inputs[0].detach())
        fea_inputs=[]
        #rnd_eps=float(np.random.rand(1)*2+1)
        epsilon=20/255
        inputs_noise=inputs+epsilon*(2*torch.rand_like(inputs)-1)
        
        inputs_noise=torch.clamp(inputs_noise,0,1)
        #outputs_adv=net(inputs_noise)
        outputs=net(inputs)
        outputs_adv=net(inputs_noise)
        loss=consist_loss_s(outputs,outputs_adv,targets,mu=18)
        
        loss.backward()
        optimizer.step()
        tea_model.update_params(net)
        tea_model.apply_shadow()

        train_loss = loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        # h_fea.remove()
        # h_fea2.remove()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.4f} \t BCE: {:.4f}'.format(
                epoch, end, len(train_loader.dataset),
                train_loss,loss.item()))
            #print(F.softmax(outputs_adv * ep, dim=1)[0].detach().cpu().numpy())
            #print('Diff is {:.4f}'.format(torch.norm(pros[index] - pros0[index]).item()))
        start = end

    acc = 100 * correct / len(train_loader.dataset)
    print('Train accuracy is {:.4f}'.format(acc))
    

    return acc, train_loss


def test(epoch):
    net.eval()
    correct=0
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pred = outputs.max(1)[1]
            correct += pred.eq(targets).sum().item()

        correct = int(correct)
        tmp_acc = 100 * correct / len(test_loader.dataset)
    print('Test acc: {:.4f}'.format(tmp_acc))
    return tmp_acc
if __name__=='__main__':
    print('==> Building model..')
    net=ResNet18(mu=1)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=3.5e-7)
    Attackers = AttackerPolymer(args.epsilon, args.num_steps, args.step_size, args.num_classes, device)

    miles = [100,200]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)
    tea_model = EMA(net)

    acc = []
    acc = np.asarray(acc, np.float32)
    

    for epoch in range(start_epoch, end_epoch):
        tr_acc, loss = train(epoch)
        te_acc = test(epoch)
        scheduler.step()
        acc = np.append(acc, epoch)
        acc = np.append(acc, tr_acc)
        acc = np.append(acc, te_acc)
        acc = np.append(acc, loss)
    # print('Higest test accuracy is:{:.4f}'.format(best_acc))
    torch.save(net.state_dict(), './model/cifar10_resnet18_lact.pkl')
    torch.save(tea_model.model.state_dict(), './model/cifar10_resnet18_tea_lact.pkl')
    acc = np.reshape(acc, (-1, 4))
    np.save('./data/cifar10_resnet18_lcat.npy', acc)
   

   

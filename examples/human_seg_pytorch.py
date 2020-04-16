import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
import math
import numpy as np
import linecache

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from trident.backend.pytorch_backend import *
from trident.layers.pytorch_blocks import  *


# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True


# C.debugging.set_computation_network_trace_level(1000)
# C.debugging.set_checked_mode(True)
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))



def _get_shape(x):
    "single object"
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    else:
        return tuple(x.size())



dataset=T.load_mnist('fashion-mnist','train',is_flatten=False)


#
# import torchvision.transforms as transforms
# transform = transforms.Compose(
# [transforms.ToTensor()])
# minst=torchvision.datasets.mnist.MNIST(root='C:/Users/Allan/.trident/datasets/minist/', train=True, transform=transform, target_transform=None, download=True)
#

# train_loader = DataLoader(
#         minst,
#         batch_size=16,
#         num_workers=0,
#         shuffle=True
#     )


num_epochs=10
minibatch_size=16



#raw_imgs, raw_labels = dataset.next_bach(minibatch_size)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*3*3 ,120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out), inplace=True)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = self.fc3(out)
        return out
#lenet_model = torch.load('lenet_model_only_pytorch_mnist.pth')
lenet_model=LeNet()
lenet_model.to(device)
#lenet_model.load_state_dict(torch.load('lenet_model_pytorch_mnist.pth'))

optimizer = optim.Adam(lenet_model.parameters(),lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
criterion=nn.CrossEntropyLoss(reduction='sum')
#
# gcd_model2 = nn.Sequential(
#     nn.Conv2d(in_channels=1,out_channels= 16, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.MaxPool2d(2,padding=1),
#     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2,padding=1),
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#     nn.Conv2d(in_channels=64, out_channels=120, kernel_size=3, padding=1),
#     nn.AdaptiveAvgPool2d(output_size=120),
#     Classifier1d(num_classes=10,is_multiselect=False, classifier_type='dense')
# )



#
# class GcdNet(nn.Module):
#     def __init__(self):
#         super(GcdNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
#         self.b1=T.GcdConv2d_Block((3, 3), 24, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         self.b2=T.GcdConv2d_Block((3, 3), 40, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         self.b2_1 = T.GcdConv2d_Block((1,1), 120, 1, auto_pad=True, activation=None, divisor_rank=0)
#         self.b2_2 = T.GcdConv2d_Block((3, 3),120, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         self.b2_3 = T.GcdConv2d_Block((1, 1), 40, 1, auto_pad=True, activation=None, divisor_rank=0)
#
#         self.b3=T.GcdConv2d_Block((3, 3), 56, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         # self.b4=T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         # self.b5=T.GcdConv2d_Block((3, 3), 104, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
#         self.b6=T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation=None)
#         self.pool=nn.AdaptiveAvgPool2d(output_size=88)
#         self.fc3 = nn.Linear(88, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.b1(out)
#         out = self.b2(out)
#         out1 = self.b2_1(out)
#         out1 = self.b2_2(out1)
#         out1 = self.b2_3(out1)
#         out=out+out1
#         out = self.b3(out)
#         # out = self.b4(out)
#         # out = self.b5(out)
#         out = self.b6(out)
#         out = self.pool(out)
#         out = out.view(out.size(0),out.size(1))
#         out = self.fc2(out)
#         out = torch.softmax(out,dim=1)
#         return out
#


def _gcd(x, y):
    gcds = []
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y // 2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds

class GCD_LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, strides, t=7, class_num=100):
        super().__init__()
        self.residual = nn.Sequential(
            T.GcdConv2d_Block_1((5, 5), num_filters=_gcd(in_channels,out_channels)[0]*t, strides=strides, activation='leaky_relu', auto_pad=True, self_norm=False, normalization='batch'),
            T.GcdConv2d_Block_1((3, 3), num_filters=out_channels, strides=1, activation=None, auto_pad=True,normalization=None))
        self.strides = strides
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        residual = self.residual(x)
        if self.strides == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual
#
class GCD_MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        self.pre =nn.Sequential(nn.Conv2d(1, 16,3, 1, padding=3), nn.BatchNorm2d(16), nn.ReLU6())
        self.stage1 = GCD_LinearBottleNeck(16, 24, 1, 1)
        self.stage3 = self._make_stage(2, 24, 32, 2, 7)
        self.stage4 = self._make_stage(2, 32, 64, 2, 9)
        self.stage5 = self._make_stage(1, 64, 96, 1, 11)
        self.stage6 = self._make_stage(1, 96, 160, 1, 13)
        self.stage7 = GCD_LinearBottleNeck(160, 320, 1, 6)
        self.conv1 =T.GcdConv2d_Block_1((3,3),num_filters=256,strides=1,activation=None,auto_pad=True,self_norm=False,normalization=None)
        self.conv2 =  T.GcdConv2d_1((1,1),num_filters=10,strides=1,self_norm=False)
    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)

        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x=torch.sigmoid(x)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(GCD_LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(GCD_LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)

class GcdNet_3(nn.Module):
    def __init__(self):
        super(GcdNet_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)#4*(2*2)
        self.b1 = T.GcdConv2d_Block_1((3, 3), 24, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 4*(2*3)
        self.b1_1 = T.GcdConv2d_Block_1((3, 3), 4*7, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  #4*7
        self.b1_2 = T.GcdConv2d_Block_1((3, 3), 84, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  #7*12
        self.b1_3 = T.GcdConv2d_Block_1((3, 3), 24, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # (2*3)*5

        self.b2_1 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,dilation=1)  # (2*3)*8
        self.b2_2 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0, dilation=3)  # (2*3)*8
        self.b2_3 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0, dilation=6)  # (2*3)*8
        #self.b2_4= T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0, dilation=12)  # (2*3)*8
        #self.b2_5= T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0, dilation=24)  # (2*3)*8


        self.b3 = T.GcdConv2d_Block_1((3, 3), 80, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 8*(2*5)
        self.b3_1 = T.GcdConv2d_Block_1((3, 3), 88, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 8*11
        self.b3_2 = T.GcdConv2d_Block_1((3, 3), 440, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 11*40
        self.b3_3 = T.GcdConv2d_Block_1((3, 3), 80, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # (2*5)*7

        self.b4_1 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0,dilation=1)  # (2*5)*12
        self.b4_2 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0,dilation=4)  # (2*5)*12
        self.b4_3 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0,dilation=8)  # (2*5)*12
        #self.b4_4 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0,dilation=16)  # (2*5)*12
        #self.b4_5 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0,dilation=32)  # (2*5)*12

        self.b5 = T.GcdConv2d_Block_1((3, 3),168 , 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # 12*(2*7)
        self.b6 = T.GcdConv2d_Block_1((5, 5), 224, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # (2*7)*16

        self.b10 = T.GcdConv2d_1((1, 1), 10, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)
        self.classifier= Classifier1d(10,classifier_type='global_avgpool')
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.b1(out)
        branch1=self.b1_1(out)
        branch1= self.b1_2(branch1)
        branch1 = self.b1_3(branch1)

        out=out+0.2*branch1

        out2_1 = self.b2_1(out)
        out2_2 = self.b2_2(out)
        out2_3 = self.b2_3(out)
        # out2_4 = self.b2_4(out)
        # out2_5 = self.b2_5(out)
        out= torch.cat([out2_1, out2_2,out2_3], 1)

        out = self.b3(out)
        branch2 = self.b3_1(out)
        branch2 = self.b3_2(branch2)
        branch2 = self.b3_3(branch2)

        out = out+0.2*branch2
        out4_1 = self.b4_1(out)
        out4_2 = self.b4_2(out)
        out4_3 = self.b4_3(out)
        # out4_4 = self.b4_4(out)
        # out4_5 = self.b4_5(out)
        out = torch.cat([out4_1, out4_2, out4_3], 1)

        out = self.b5(out)
        out = self.b6(out)
        out = self.b10(out)
        out = self.classifier(out)
        return out


#gcd_model = torch.load('gcd_model_only_pytorch_mnist.pth')
gcd_model=GcdNet_3()

gcd_model.to(device)
flops_gcd_model = T.calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model', flops_gcd_model))




#gcd_model.load_state_dict(torch.load('gcd_model_pytorch_mnist.pth'))
for mbs,(input, target)  in enumerate(dataset):
    input, target =torch.from_numpy(input),torch.from_numpy(target)
    test_input, test_target = Variable(input).to(device), Variable(target).to(device)
    gcd_model(test_input)
    break

print(gcd_model)
for i,para in enumerate(gcd_model.parameters()):
    print('{0} {1} {2}'.format(i,para.name,_get_shape(para)))

gcd_optimizer = optim.Adam(gcd_model.parameters(),lr=0.001)
gcd_mse_criterion=nn.MSELoss(reduction='mean')
gcd_ce_criterion=nn.CrossEntropyLoss(reduction='sum')
#flops_model = calculate_flops(model)
#print('flops_model:{0}'.format(flops_model))



flops_lenet_model = T.calculate_flops(lenet_model)
print('flops_{0}:{1}'.format('lenet_model', flops_lenet_model))

flops_gcd_model = T.calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model', flops_gcd_model))

#f = codecs.open('model_log_cifar100.txt', 'a', encoding='utf-8-sig')
#model = Function.load('Models/model5_cifar100.model')
os.remove('model_log_fmnist_test3d.txt')
f = codecs.open('model_log_fmnist_test3d.txt', 'a', encoding='utf-8-sig')
losses=[]
metrics=[]

gcd_losses=[]
gcd_metrics=[]
print('epoch start')
for epoch in range(num_epochs):

    while mbs<=2000:
        for mbs,(input, target) in enumerate(dataset):
            input, target =torch.from_numpy(input),torch.from_numpy(target)
            input, target = Variable(input).to(device), Variable(target).to(device)
            lenet_output = lenet_model(input)

            lenet_loss=criterion(lenet_output,target)

            accu = 1-np.mean(np.not_equal(np.argmax(lenet_output.cpu().detach().numpy(), -1).astype(np.int64), target.cpu().detach().numpy().astype(np.int64)))
            losses.append(lenet_loss.item())
            metrics.append(accu)


            optimizer.zero_grad()
            lenet_loss.backward()
            optimizer.step()

            if mbs>=0:
                inputs, targets_a, targets_b, lam = mixup_data(input, target, 1, True)
                gcd_optimizer.zero_grad()
                gcd_output = gcd_model(input)
                #print(gcd_output.cpu().detach().numpy()[:2])
                #gcd_loss =gcd_ce_criterion(gcd_output,target)#+ gcd_mse_criterion(gcd_output, lenet_output)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                gcd_loss = loss_func(gcd_ce_criterion, gcd_output)
                # if epoch>0 and mbs>100:
                #     result=gcd_output.cpu().detach().numpy()
                #     print(result)
                gcd_accu = 1 - np.mean(np.not_equal(np.argmax(gcd_output.cpu().detach().numpy(), -1).astype(np.int64),
                                                    target.cpu().detach().numpy().astype(np.int64)))
                gcd_losses.append(gcd_loss.item())
                gcd_metrics.append(gcd_accu)
                gcd_loss.backward( )
                gcd_optimizer.step()


            if mbs % 50 == 0:
                print("Baseline:     Epoch: {}/{} ".format(epoch + 1, num_epochs),
                      "Step: {} ".format(mbs),
                      "Loss: {:.4f}...".format(np.array(losses).mean()),
                      "Accuracy:{:.3%}...".format(np.array(metrics).mean()))
                f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('lanet_model', 0.01, epoch, mbs + 1, np.array(losses).mean(), np.array(metrics).mean())])


                losses=[]
                metrics=[]
                if mbs >=0:
                    print("Gcd_Conv    Epoch: {}/{} ".format(epoch + 1, num_epochs), "Step: {} ".format(mbs),
                          "Loss: {:.4f}...".format(np.array(gcd_losses).mean()),
                          "Accuracy:{:.3%}...".format(np.array(gcd_metrics).mean()))
                    f.writelines([
                        'model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('gcd_model', 0.01, epoch, mbs + 1, np.array(gcd_losses).mean(), np.array(gcd_metrics).mean())])

                gcd_losses = []
                gcd_metrics = []
            if (mbs+1)%100==0:
                # torch.save(lenet_model.state_dict(),'lenet_model_pytorch_mnist_1.pth' )
                # torch.save(gcd_model.state_dict(), 'gcd_model_pytorch_mnist_1.pth')
                torch.save(lenet_model, 'lenet_model_only_pytorch_fmnist.pth')
                torch.save(gcd_model, 'gcd_model_only_pytorch_fmnist.pth')

            mbs += 1

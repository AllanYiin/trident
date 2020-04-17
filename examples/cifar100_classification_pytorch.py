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
from  logging import *

from trident.backend.pytorch_backend import *
from trident.layers.pytorch_blocks import  *
from trident.layers.pytorch_normalizations import  *

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




num_epochs=50
minibatch_size=8


dataset = T.load_cifar('cifar100', 'train', is_flatten=False)
dataset.minibatch_size=minibatch_size



class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels, in_channels * t, 1), nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual

class MobileNetV2(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(32), nn.ReLU6())
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(2, 24, 32, 2, 6)
        self.stage4 = self._make_stage(2, 32, 64, 2, 6)
        self.stage5 = self._make_stage(1, 64, 96, 1, 6)
        self.stage6 = self._make_stage(1, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        self.conv1 = nn.Sequential(nn.Conv2d(320, 1280, 1), nn.BatchNorm2d(1280), nn.ReLU6())
        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x=torch.softmax(x,dim=1)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)


def mobilenetv2():
    return MobileNetV2()


#mobilenetv2_model = torch.load('mobilenetv2_cifar100_model_only_pytorch_mnist.pth')
mobilenetv2_model=mobilenetv2()
#print_network(mobilenetv2_model)
mobilenetv2_model.to(device)
#lenet_model.load_state_dict(torch.load('lenet_model_pytorch_mnist.pth'))
flops_mobilenetv2_model = calculate_flops(mobilenetv2_model)
print('flops_{0}:{1}'.format('baseline flops_mobilenetv2_model', flops_mobilenetv2_model))

for i,para in enumerate(mobilenetv2_model.named_parameters()):
    print('{0} {1} {2}'.format(i,para[0],_get_shape(para[1])))

for i,(test_input, _ ) in  enumerate(dataset):
    test_input = torch.from_numpy(test_input)
    test_input= Variable(test_input)
    test_input=test_input.to(device)
    mobilenetv2_model(test_input)
    break


flops_mobilenetv2_model = calculate_flops(mobilenetv2_model)
print('flops_{0}:{1}'.format('baseline flops_mobilenetv2_model', flops_mobilenetv2_model))


optimizer = optim.Adam(mobilenetv2_model.parameters(),lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
criterion=nn.CrossEntropyLoss(reduction='mean')



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
            T.GcdConv2d_Block_1((3, 3), num_filters=_gcd(in_channels,out_channels)[0]*t, strides=strides, activation='leaky_relu6', auto_pad=True, self_norm=True, normalization=None,use_bias=False),
            T.GcdConv2d_Block_1((3, 3), num_filters=out_channels, strides=1, activation='leaky_relu6', auto_pad=True,normalization=None,use_bias=False))
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
        self.pre =nn.Sequential(nn.Conv2d(3, 32,3, 1, padding=1), nn.BatchNorm2d(32), T.LeakyReLU())
        self.stage1 = GCD_LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 5)
        self.stage3 = self._make_stage(2, 24, 32, 2, 7)
        self.stage4 = self._make_stage(2, 32, 64, 2, 11)
        self.stage5 = self._make_stage(1, 64, 96, 1, 13)
        self.stage6 = self._make_stage(1, 96, 160, 1, 17)
        self.stage7 = GCD_LinearBottleNeck(160, 320, 1, 23)
        self.conv1 =T.GcdConv2d_Block_1((3,3),num_filters=256,strides=1,activation='leaky_relu6',auto_pad=True,self_norm=True,normalization=None)
        self.conv2 =  T.GcdConv2d_1((1,1),num_filters=100,strides=1,self_norm=False)
    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x=x.sigmoid()
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(GCD_LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(GCD_LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)




def gcd_mobilenetv2():
    return GCD_MobileNetV2()





#gcd_model = torch.load('gcd_cifar100_model_only_pytorch_mnist.pth')

gcd_model=gcd_mobilenetv2()
#print_network(gcd_model)
gcd_model.to(device)

for i,para in enumerate(gcd_model.parameters()):
    print('{0} {1} {2}'.format(i,para.name,_get_shape(para)))


for i,(test_input, _ ) in  enumerate(dataset):
    test_input = torch.from_numpy(test_input)
    test_input= Variable(test_input)
    test_input=test_input.to(device)
    result=gcd_model(test_input)
    break

#gcd_model.load_state_dict(torch.load('gcd_model_pytorch_mnist.pth'))
import collections
weight_dict=collections.OrderedDict()
weight_dict_current=collections.OrderedDict()
flops_gcd_model = calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model v2', flops_gcd_model))


flops_gcd_model = calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model v2', flops_gcd_model))
for i,para in enumerate(gcd_model.named_parameters()):
    weight_dict_current[para[0]] = para[1].data.cpu().detach().numpy()
    print('{0} {1} {2} {3}'.format(i,para[0],_get_shape(para[1]),para[1].requires_grad))

gcd_optimizer = optim.Adam(gcd_model.parameters(),lr=0.05, betas=(0.5, 0.999), eps=1e-08, weight_decay=0.005)
#gcd_mse_criterion=nn.MSELoss(reduction='mean')
gcd_ce_criterion=nn.CrossEntropyLoss(reduction='mean')
gcd_null_criterion=nn.NLLLoss(reduction='mean')
#flops_model = calculate_flops(model)
#print('flops_model:{0}'.format(flops_model))



flops_lenet_model = calculate_flops(mobilenetv2_model)
print('flops_{0}:{1}'.format('mobilenet v2', flops_lenet_model))

flops_gcd_model = calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model', flops_gcd_model))

print('{0:.3%}'.format(flops_gcd_model/flops_lenet_model))

#f = codecs.open('model_log_cifar100.txt', 'a', encoding='utf-8-sig')
#model = Function.load('Models/model5_cifar100.model')
#os.remove('model_log_cifar100_baseline_lr2.txt')
f = codecs.open('model_log_cifar100_test3d.txt', 'a', encoding='utf-8-sig')
losses=[]
metrics=[]

gcd_losses=[]
gcd_metrics=[]

print('epoch start')
for epoch in range(num_epochs):
    mbs = 0
    for mbs ,(input, target) in enumerate(dataset):
        input, target = torch.from_numpy(input), torch.from_numpy(target)
        input, target = Variable(input).to(device), Variable(target).to(device)
        mobilenetv2_output = mobilenetv2_model(input)

        mobilenetv2_loss=criterion(mobilenetv2_output,target)

        accu = 1-np.mean(np.not_equal(np.argmax(mobilenetv2_output.cpu().detach().numpy(), -1).astype(np.int64), target.cpu().detach().numpy().astype(np.int64)))
        losses.append(mobilenetv2_loss.item())
        metrics.append(accu)


        optimizer.zero_grad()
        mobilenetv2_loss.backward( retain_graph=True)
        optimizer.step()

        #if mbs>=0:
        gcd_optimizer.zero_grad()
        gcd_output = gcd_model(input)
        gcd_loss =gcd_null_criterion(F.log_softmax(gcd_output,dim=1),target)+gcd_ce_criterion(gcd_output,target)#+ gcd_mse_criterion(gcd_output, mobilenetv2_output)
        if (mbs + 1) % 100 == 0:
            weight_dict=weight_dict_current.copy()
            for i, para in enumerate(gcd_model.named_parameters()):
                weight_dict_current[para[0]]=para[1].data.cpu().detach().numpy()
                print('{0} {1} {2}'.format(i, para[0], _get_shape(para[1])))
            for k,v in weight_dict_current.items():
                print('{0}   mean: {1:.5f} max: {2:.5f} min: {3:.5f}  diff:{4:.3%}  '.format(k,v.mean(), v.max(),v.min(),np.abs(v-weight_dict[k]).mean()/np.abs(weight_dict[k]).mean()))
            result = gcd_output.cpu().detach().numpy()
            print('mean: {0:} max: {1} min: {2}'.format(result.mean(), result.max(), result.min()))
        gcd_accu = 1 - np.mean(np.not_equal(np.argmax(gcd_output.cpu().detach().numpy(), -1).astype(np.int64),
                                            target.cpu().detach().numpy().astype(np.int64)))
        gcd_losses.append(gcd_loss.item())
        gcd_metrics.append(gcd_accu)
        gcd_loss.backward( )
        gcd_optimizer.step()


        if mbs % 20 == 0:
            print("MobileNetv2:     Epoch: {}/{} ".format(epoch + 1, num_epochs),
                  "Step: {} ".format(mbs),
                  "Loss: {:.4f}...".format(np.array(losses).mean()),
                  "Accuracy:{:.3%}...".format(np.array(metrics).mean()))
            f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('mobilenetv2_model', 0.01, epoch, mbs + 1, np.array(losses).mean(), np.array(metrics).mean())])


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
            torch.save(mobilenetv2_model, 'mobilenetv2_cifar100_model_only_pytorch_mnist.pth')
            torch.save(gcd_model, 'gcd_cifar100_model_only_pytorch_mnist.pth')



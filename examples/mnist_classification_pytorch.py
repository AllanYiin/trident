import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'pytorch'
from trident import backend as T
import math
import numpy as np
import linecache

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from trident.backend.pytorch_backend import *
from trident.layers.pytorch_blocks import  *

# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def calculate_flops(x):
    flops = 0
    for p in x.parameters:
        flops += p.value.size
    return flops


data = T.load_cifar('cifar100', 'train', is_flatten=False)
dataset = T.Dataset('cifar100')
dataset.mapping(data=data[0], labels=data[1], scenario='train')


num_epochs=5
minibatch_size=32



raw_imgs, raw_labels = dataset.next_bach(minibatch_size)



# part=conv3x3(3, 16, 1)
# part=nn.BatchNorm2d(16)(part)
# part=nn.Sequential(conv3x3(3, 16, 1),nn.BatchNorm2d(16))(part)
# part=conv3x3(16, 32, 1)(part)
# class baselineNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(baselineNet, self).__init__()
#         self.conv1 = conv3x3(3, 16, 1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = conv3x3(16, 32, 1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = conv3x3(32, 64, 1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.conv4 = conv3x3(64, 128, 1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.conv5 = conv3x3(128, 256, 1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = conv3x3(256, 512, 1)
#         self.bn6 = nn.BatchNorm2d(512)
#         self.conv7 = conv3x3(512, 512, 1)
#         self.bn7 = nn.BatchNorm2d(512)
#
#         self.conv_xy = conv1x1(512, 2, 1)
#         self.conv_wh = conv1x1(512, 2, 1)
#         self.conv_obj = conv1x1(512, 1, 1)
#         self.conv_class = conv1x1(512, num_classes, 1)
#
#         self.leakyrelu = nn.LeakyReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.leakyrelu(x)
#         x = self.maxpool(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.leakyrelu(x)
#         x = self.maxpool(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.leakyrelu(x)
#         x = self.maxpool(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.leakyrelu(x)
#         x = self.maxpool(x)
#
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.leakyrelu(x)
#
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.leakyrelu(x)
#
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.leakyrelu(x)
#
#         bbox_xy = self.conv_xy(x)
#         bbox_xy = self.sigmoid(bbox_xy)
#
#         bbox_wh = self.conv_wh(x)
#
#         bbox_obj = self.conv_obj(x)
#         bbox_obj = self.sigmoid(bbox_obj)
#
#         bbox_class = self.conv_class(x)
#         bbox_class = self.softmax(bbox_class)
#
#         p = torch.cat((bbox_xy, bbox_wh, bbox_obj, bbox_class), 1)
#
#         return p
#
#
# tiny_yolo=baselineNet(10)
# model=tiny_yolo.to(device)

model1 = nn.Sequential(
    nn.Conv2d(3, 32, (3, 3), 1, padding=1, bias=False),
    T.Conv2d_Block((3, 3), 64, 1, auto_pad=True, activation='relu6', normalization='instance'),
    T.Conv2d_Block((3, 3), 64, 2, auto_pad=True, activation='relu6', normalization='instance'),
    T.Conv2d_Block((3, 3), 128, 1, auto_pad=True, activation='relu6', normalization='instance',dropout_rate=0.2),
    T.Conv2d_Block((3, 3), 128, 2, auto_pad=True, activation='relu6', normalization='instance'),
    T.Conv2d_Block((3, 3), 256, 1, auto_pad=True, activation='relu6', normalization='instance'),
    T.ShortCut2d({
                        'left': [
                            nn.Conv2d(256, 64, (1, 1), 1, padding=0, bias=False),
                            nn.Conv2d(64, 256, (3, 3), 1, padding=1, bias=False),],
                        'right': [nn.Conv2d(256, 256, (1, 1), 1, padding=0, bias=False) ]}),
    T.Conv2d_Block((3, 3), 256, 2, auto_pad=True, activation='relu6', normalization='instance'),
    Classifier1d(num_classes=100,is_multiselect=False, classifier_type='dense')
)




model = nn.Sequential(
    nn.Conv2d(3, 32, (3, 3), 1, padding=1, bias=False),
    T.GcdConv2d_Block((3, 3), 48, 1, auto_pad=True, activation='relu6',divisor_rank=0),
    T.GcdConv2d_Block((3, 3), 80, 2, auto_pad=True, activation='relu6', divisor_rank=0),
    T.GcdConv2d_Block((3, 3), 112, 1, auto_pad=True, activation='relu6', divisor_rank=0,dropout_rate=0.2),
    T.GcdConv2d_Block((3, 3), 176, 2, auto_pad=True, activation='relu6', divisor_rank=0),
    T.GcdConv2d_Block((3, 3), 208, 1, auto_pad=True, activation='relu6', divisor_rank=0),
    T.ShortCut2d({
                        'left': [
                            T.GcdConv2d((1, 1), 96, 1, auto_pad=True),
                            T.GcdConv2d((3, 3), 256, 1, auto_pad=True)],
                        'right': [T.GcdConv2d((1, 1), 256, 1, auto_pad=True) ]}),
    T.GcdConv2d_Block((3, 3), 352, 2, auto_pad=True, activation='relu6'),
    T.GcdConv2d_Block((3, 3), 440, 1, auto_pad=True, activation=None),
    Classifier1d(num_classes=100,is_multiselect=False, classifier_type='dense')
)
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
criterion=nn.CrossEntropyLoss(reduction='sum')

#flops_model = calculate_flops(model)
#print('flops_model:{0}'.format(flops_model))



#f = codecs.open('model_log_cifar100.txt', 'a', encoding='utf-8-sig')
#model = Function.load('Models/model5_cifar100.model')

losses=[]
metrics=[]
print('epoch start')
for epoch in range(num_epochs):
    mbs = 0
    while mbs<=1000:
        input, target  = dataset.next_bach(64)
        input, target = torch.from_numpy(input), torch.from_numpy(target)
        input, target = Variable(input).to(device), Variable(target).to(device)
        output = model(input)
        loss=criterion(output,target)
        accu = 1-np.mean(np.not_equal(np.argmax(output.cpu().detach().numpy(), -1).astype(np.int64), target.cpu().detach().numpy().astype(np.int64)))
        losses.append(loss.item())
        metrics.append(accu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if mbs % 50 == 0:
            print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
                  "Step: {} ".format(mbs),
                  "Loss: {:.4f}...".format(np.array(losses).mean()),
                  "Accuracy:{:.3%}...".format(np.array(metrics).mean()))
            losses=[]
            metrics=[]

        mbs += 1

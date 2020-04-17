import os
import sys
import codecs
import glob
import PIL.Image as Image
import cv2
import random
import math
import datetime
import time
import numpy as np
import linecache
import matplotlib
import matplotlib.pyplot as plt
import pylab
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
os.environ['TRIDENT_BACKEND'] = 'pytorch'

import trident as T
from trident import *
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



def _get_shape(x):
    "single object"
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    else:
        return tuple(x.size())

imgs_path = glob.glob('C:/Users/Allan/PycharmProjects/DeepBelief_Course4_Examples/Data/ex05_train/imgs/' + '*.jpg')
mks_path = ['C:/Users/Allan/PycharmProjects/DeepBelief_Course4_Examples/Data/ex05_train/masks/' + os.path.basename(i)[:-4] + '.png' for i in imgs_path]
# img_data=[T.read_image(im_path)  for im_path in imgs_path]
# mask_data=[T.read_mask(im_path)  for im_path in mks_path]
# dataset=T.DataProvider('seg',data=img_data,masks=mask_data)


# 數據增強處理函數

#圖片轉向量
def img2array(img: Image):
    arr = np.array(img).astype(np.float32)
    arr=arr.transpose(2, 0, 1) #轉成CHW
    arr=np.ascontiguousarray(arr)
    return arr[::-1] #顏色排序為BGR

#向量轉圖片
def array2img(arr: np.ndarray):
    arr =arr[::-1]#轉成RGB
    sanitized_img = np.maximum(0, np.minimum(255, np.transpose(arr, (1, 2, 0))))#轉成HWC
    img = Image.fromarray(sanitized_img.astype(np.uint8))
    return img

# 隨機裁切
def random_crop(image, mask):
    min_len = min(image.width, image.height)
    # print(min_len)
    scale = 1.
    if min_len < 128 or min_len > 128 * 1.5:
        scale = np.random.choice(np.arange(128. / min_len, 1.5 * 128. / min_len, 0.1))
    image = image.resize((int(image.width * scale) + 2, int(image.height * scale) + 2), Image.ANTIALIAS)
    image = img2array(image)
    mask = mask.resize((int(mask.width * scale) + 2, int(mask.height * scale) + 2), Image.NEAREST)
    mask = img2array(mask)
    # print('image:{0}'.format(image.shape))
    # print('mask:{0}'.format(mask.shape))
    offset_x = random.choice(range(0, image.shape[2] - 128))
    offset_y = random.choice(range(0, image.shape[1] - 128))
    # print('offset_x:{0}'.format(offset_x))
    # print('offset_y:{0}'.format(offset_y))
    image = image[:, offset_y:offset_y + 128, offset_x:offset_x + 128]
    mask = mask[:, offset_y:offset_y + 128, offset_x:offset_x + 128]
    # print('crop image:{0}'.format(image.shape))
    # print('crop mask:{0}'.format(mask.shape))
    return image, mask[0, :, :]


# 隨機加入標準常態分配的噪聲
def add_noise(image):
    noise = np.random.standard_normal(image.shape) * np.random.choice(np.arange(-15, 15))
    image = np.clip(image + noise, 0, 255)
    return image


# 調整明暗
def adjust_gamma(image, gamma=1.8):
    image = image.transpose([1, 2, 0])
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    cv2.LUT(image.astype(np.uint8), table)
    image = image.transpose([2, 0, 1])
    return image


# 模糊
def adjust_blur(image, gamma=1.8):
    image = image.transpose([1, 2, 0])
    image = cv2.blur(image, (3, 3))
    image = image.transpose([2, 0, 1])
    return image



#把圖片並列列印出來
def tile_rgb_images(pred, label, truth, row=3, col=3):
    fig = pylab.gcf()
    fig.set_size_inches(20, 18)

    fig.set_size_inches(col * 2, row * 2)
    pylab.clf()
    pylab.ioff()  # is not None:
    prefix = str(datetime.datetime.fromtimestamp(time.time())).replace(' ', '').replace(':', '').replace('-',
                                                                                                         '').replace(
        '.', '')
    for m in range(row * col):
        pylab.subplot(row, col, m + 1)
        if m % 3 == 0:
            img = array2img(truth[int(m/ 3)]*pred[int(m / 3)]+np.ones((3,128,128))*120*(1-pred[int(m / 3)]))
        elif m % 3 == 1:

            img = array2img(truth[int(m/ 3)]*label[int((m - 1) / 3)]+np.ones((3,128,128))*120*(1-label[int(m / 3)]))
        else:
            img = array2img(truth[int((m - 2) / 3)])
        pylab.imshow(img, interpolation="nearest", animated=True)
        pylab.axis("off")
    filename = 'Results/gcd_seg_{0}.png'.format(prefix)
    pylab.savefig(filename, bbox_inches='tight')


idx = 0
idxs = np.arange(len(imgs_path))
np.random.shuffle(idxs)


def get_next_minibatch(minibatch_size=8, is_train=True):
    global idx, idxs
    features = []
    masks = []
    sizes = []
    while len(features) < minibatch_size:
        im, mk = random_crop(Image.open(imgs_path[idxs[idx]]), Image.open(mks_path[idxs[idx]]))

        # print(im.shape),print(mk.shape)
        if is_train:
            if random.randint(0, 10) % 3 == 0:
                im = add_noise(im)

            if random.randint(0, 10) % 5 <= 1:  # 水平翻轉
                im = im[:, :, ::-1]
                mk = mk[:, ::-1]

            if random.randint(0, 10) % 3 == 0:
                im = adjust_blur(im)

            if random.randint(0, 10) % 2 == 0:  # 明暗
                gamma = np.random.choice(np.arange(0.6, 1.4, 0.05))
                img = adjust_gamma(im, gamma)

        mk[mk > 0] = 1

        # print(decode_mk.shape)

        features.append(im / 255.0)
        masks.append(mk)
        idx += 1
        if idx >= len(imgs_path) - 1:
            idx = 0
            np.random.shuffle(idxs)
    return np.asarray(features).astype(np.float32), np.asarray(masks).astype(np.int64)


num_epochs=10
minibatch_size=4



class GcdNet_3(nn.Module):
    def __init__(self):
        super(GcdNet_3, self).__init__()
        self.conv1 = Conv2d((3, 3), 16, 1, auto_pad=True, activation='leaky_relu')
        self.b1 = T.GcdConv2d_Block_1((5, 5), 24, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 4*(2*3)
        self.b1_1 = T.GcdConv2d_Block_1((3, 3), 4 * 7, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 4*7
        self.b1_2 = T.GcdConv2d_Block_1((3, 3), 84, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 7*12
        self.b1_3 = T.GcdConv2d_Block_1((3, 3), 24, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # (2*3)*5

        self.b2_1 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0, dilation=1)  # (2*3)*8
        self.b2_2 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,  dilation=3)  # (2*3)*8
        self.b2_3 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,  dilation=6)  # (2*3)*8
        self.b2_4 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0, dilation=12)  # (2*3)*8
        self.b2_5 = T.GcdConv2d_Block_1((3, 3), 40, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,  dilation=24)  # (2*3)*8

        self.b3 = T.GcdConv2d_Block_1((3, 3), 72, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 8*9
        self.b3_1 = T.GcdConv2d_Block_1((3, 3), 99, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 9*11
        self.b3_2 = T.GcdConv2d_Block_1((3, 3), 264, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)  # 3*11*(8
        self.b3_3 = T.GcdConv2d_Block_1((3, 3), 72, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # 8*9

        self.b4_1 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,dilation=1)  # (2*5)*12
        self.b4_2 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,dilation=4)  # (2*5)*12
        self.b4_3 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0,dilation=8)  # (2*5)*12
        self.b4_4 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0, dilation=16)  # (2*5)*12
        self.b4_5 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0, dilation=32)  # (2*5)*12
        self.b4_6 = T.GcdConv2d_Block_1((3, 3), 104, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0, dilation=60)  # (2*5)*12

        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.se1 = T.GcdConv2d_Block_1((1, 1), 84 , 1, auto_pad=True, activation='relu', divisor_rank=0)
        self.se2 = T.GcdConv2d_Block_1((1, 1), 624, 1, auto_pad=True, activation='sigmoid', divisor_rank=0)

        self.b5 = T.GcdConv2d_Block_1((3, 3), 143, 1, auto_pad=True, activation='leaky_relu',divisor_rank=0)  # 12*(2*7)

        self.b6 = T.GcdConv2d_Block_1((5, 5), 242, 1, auto_pad=True, activation='leaky_relu', divisor_rank=0)
        self.att =Conv2d((1, 1), 242, 1, auto_pad=True, activation='sigmoid')  # (2*7)*16

        self.b10 = Conv2d((1, 1), 1, 1, auto_pad=True, activation='sigmoid')


    def forward(self, x):
        out = self.conv1(x)
        out = self.b1(out)
        branch1 = self.b1_1(out)
        branch1 = self.b1_2(branch1)
        branch1 = self.b1_3(branch1)

        out = out + 0.2 * branch1

        out2_1 = self.b2_1(out)
        out2_2 = self.b2_2(out)
        out2_3 = self.b2_3(out)
        out2_4 = self.b2_4(out)
        out2_5 = self.b2_5(out)
        out = torch.cat([out2_1, out2_2, out2_3, out2_4, out2_5], 1)

        out = self.b3(out)
        branch2 = self.b3_1(out)
        branch2 = self.b3_2(branch2)
        branch2 = self.b3_3(branch2)

        out = out + 0.2 * branch2
        out4_1 = self.b4_1(out)
        out4_2 = self.b4_2(out)
        out4_3 = self.b4_3(out)
        out4_4 = self.b4_4(out)
        out4_5 = self.b4_5(out)
        out4_6 = self.b4_6(out)
        out = torch.cat([out4_1, out4_2, out4_3, out4_4, out4_5,out4_6], 1)

        se=self.global_avgpool(out)
        se=self.se1(se)
        se=self.se2(se)
        out=out*se

        out = self.b5(out)
        out = self.b6(out)
        att=out.mean(-1,True).mean(-2,True)
        att=self.att(att)
        out=out*att
        out = self.b10(out)
        return out


gcd_model = torch.load('Models/gcd_human_seg_pytorch_se.pth')
#gcd_model=GcdNet_3()

gcd_model.to(device)





test_input, test_target= get_next_minibatch(minibatch_size,True)
test_input, test_target =torch.from_numpy(test_input),torch.from_numpy(test_target)
test_input, test_target = Variable(test_input).to(device), Variable(test_target).to(device)
gcd_model(test_input)

T.summary(gcd_model,(3,128,128))



import collections
# weight_dict=collections.OrderedDict()
# weight_dict_current=collections.OrderedDict()
# for i,para in enumerate(gcd_model.named_parameters()):
#     if para[1].requires_grad==True:
#         weight_dict_current[para[0]] = para[1].data.cpu().detach().numpy()
#         #print('{0} {1} {2} {3}'.format(i,para[0],_get_shape(para[1]),para[1].requires_grad))
#


def adjust_learning_rate(optimizer,lr=0.0001):
    """Sets the learning rate: milestone is a list/tuple"""
    print('learning rate : {0}'.format(lr))
    optimizer.param_groups[0]['lr'] = lr
    return lr

gcd_optimizer = optim.Adam(gcd_model.parameters(),lr=1e-4)
gcd_ce_criterion=nn.CrossEntropyLoss(reduction='mean')
gcd_dice=T.DiceLoss(ignore_index=0)
gcd_mse_criterion=nn.MSELoss(reduction='mean')
gcd_focal_criterion=T.FocalLoss(ignore_index=0)
gce_iou_criterion=T.SoftIoULoss(2)
gce_lovas_criterion=T.LovaszSoftmax(reduction='mean')
#flops_model = calculate_flops(model)
#print('flops_model:{0}'.format(flops_model))





flops_gcd_model = T.calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model', flops_gcd_model))

#f = codecs.open('model_log_cifar100.txt', 'a', encoding='utf-8-sig')
#model = Function.load('Models/model5_cifar100.model')
#os.remove('model_log_fmnist_test3d.txt')
f = codecs.open('model_log_human_seg_se.txt', 'a', encoding='utf-8-sig')
losses=[]
metrics=[]

gcd_losses=[]
gcd_metrics=[]
trn_intersection= 0
trn_union = 0
tot=0
print('epoch start')
for epoch in range(num_epochs):
    mbs=0
    while mbs<=2000:
        raw_features, raw_target= get_next_minibatch(minibatch_size,True)
        input, target =torch.from_numpy(raw_features),torch.from_numpy(raw_target)
        input, target = Variable(input).to(device), Variable(target).to(device)


        gcd_optimizer.zero_grad()
        gcd_output = gcd_model(input)
        gcd_output2=torch.cat([1-gcd_output,gcd_output], 1)

        output = T.to_numpy(gcd_output2)

        gcd_loss = gcd_focal_criterion(gcd_output2, target) +gce_iou_criterion(gcd_output2,target)+gcd_mse_criterion(gcd_output,T.make_one_hot(target,2)[:,1:2,:,:])+ gcd_dice(gcd_output2, target).sum()

        gcd_accu =T.accuracy(gcd_output,target)
        gcd_losses.append(gcd_loss.item())
        gcd_metrics.append(T.to_numpy(gcd_accu))
        gcd_loss.backward( )
        gcd_optimizer.step()

        pred = np.argmax(output, 1)
        mask = raw_target
        intersection = np.sum(np.equal(pred, mask) * np.greater(mask, 0))
        union = np.sum(np.greater(pred + mask, 0))
        trn_intersection += intersection
        trn_union += union


        if mbs % 10 == 0:

            print("Gcd_Conv    Epoch: {}/{} ".format(epoch + 1, num_epochs), "Step: {} ".format(mbs),
                  "Loss: {:.4f}...".format(np.array(gcd_losses).mean()),
                  "IOU:{:.3%}...".format(trn_intersection/trn_union),
                  "mean:{:.3%}...".format(T.to_numpy(gcd_output).mean()))
            f.writelines([
                'model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('gcd_model', 0.01, epoch, mbs + 1, np.array(gcd_losses).mean(), trn_intersection/trn_union)])

            gcd_losses = []
            gcd_metrics = []
            trn_intersection = 0
            trn_union = 0
        if (mbs+1)%10==0:
            # torch.save(lenet_model.state_dict(),'lenet_model_pytorch_mnist_1.pth' )
            # torch.save(gcd_model.state_dict(), 'gcd_model_pytorch_mnist_1.pth')

            torch.save(gcd_model, 'Models/gcd_human_seg_pytorch_se.pth')
            pred = np.greater(T.to_numpy(gcd_output),0.5)
            tile_rgb_images(pred, raw_target, raw_features * 255., row=3, col=3)
        # if (mbs ) % 100 == 0:
        #     weight_dict = weight_dict_current.copy()
        #     for i, para in enumerate(gcd_model.named_parameters()):
        #         if para[1].requires_grad==True:
        #             weight_dict_current[para[0]] = para[1].data.cpu().detach().numpy()
        #     #     print('{0} {1} {2}'.format(i, para[0], _get_shape(para[1])))
        #     for k, v in weight_dict_current.items():
        #         print('{0}   mean: {1:.5f} max: {2:.5f} min: {3:.5f}  diff:{4:.3%}  '.format(k, v.mean(), v.max(),
        #                                                                                      v.min(), np.abs(
        #                 v - weight_dict[k]).mean() / np.abs(weight_dict[k]).mean()))
        #     result = T.to_numpy(gcd_output)
        #     print('mean: {0:} max: {1} min: {2}'.format(result.mean(), result.max(), result.min()))

        if tot==100:
            adjust_learning_rate(gcd_optimizer,2e-6)
        elif tot==200:
            adjust_learning_rate(gcd_optimizer, 3e-6)
        elif tot==300:
            adjust_learning_rate(gcd_optimizer, 4e-6)
        elif tot==400:
            adjust_learning_rate(gcd_optimizer, 5e-6)
        elif tot==500:
            adjust_learning_rate(gcd_optimizer, 5e-6)
        elif tot>500 and tot %250==0:
            adjust_learning_rate(gcd_optimizer, 5e-4* math.pow(0.9,(tot//200)+1))
        mbs += 1
        tot+=1

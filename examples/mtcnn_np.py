import numpy as np
import scipy.signal as signal
import scipy as sp
import cv2
import math
import torch
import collections
import PIL
import PIL.Image as Image
from enum import Enum, unique

det1=np.load('det1.npy',allow_pickle=True, encoding='latin1').item()
det2=np.load('det2.npy',allow_pickle=True, encoding='latin1').item()
det3=np.load('det3.npy',allow_pickle=True, encoding='latin1').item()
#
# det1_1=torch.load('pnet.pkl')
# det1_2=collections.OrderedDict()
# for k,v in det1_1.items():
#     det1_2[k]=v.data.cpu().detach().numpy()
# np.save('det1.npy',det1_2)
#
# det2_1=torch.load('rnet.pkl')
# det2_2=collections.OrderedDict()
# for k,v in det2_1.items():
#     det2_2[k]=v.data.cpu().detach().numpy()
# np.save('det2.npy',det2_2)
# det3_1=torch.load('onet.pkl')
# det3_2=collections.OrderedDict()
# for k,v in det3_1.items():
#     det3_2[k]=v.data.cpu().detach().numpy()
# np.save('det3.npy',det3_2)


def convolve(inp,kernal):
    output=np.zeros((inp.shape[0]-kernal.shape[0]+1,inp.shape[1]-kernal.shape[1]+1))
    for x in range(inp.shape[1]-kernal.shape[1]+1):  # Loop over every pixel of the image
        for y in range(inp.shape[0]-kernal.shape[0]+1):
            output[y, x] = (kernal * inp[y:y + kernal.shape[0], x:x + kernal.shape[1]]).sum()
    return output

# inp=np.reshape(np.arange(25),(5,5))
# kernal=np.reshape(np.arange(9),(3,3))
# result=convolve(inp,kernal)
# kernal=np.flipud(np.fliplr(kernal))
# result1=signal.convolve(inp,kernal,mode='valid')
# result2=signal.convolve2d(inp,kernal,mode='valid')

def conv(x, w,bias, strides,padding):
    y = []
    for i in range(x.shape[0]):
        _y = []
        for j in range(w.shape[0]):
            __y = []
            for k in range(w.shape[1]):
                __y.append(signal.convolve(x[i, k], np.flipud(np.fliplr(w[j, k])),mode=padding)[::strides[0], ::strides[1]])
            _y.append(np.sum(np.stack(__y, axis=-1), axis=-1))
        y.append(_y)
    y = np.array(y)
    if bias is not None:
        y = y + np.reshape(bias, (1, -1, 1, 1))
    return y



def prelu(x,w):
    shape=list((1,)*len(x.shape))
    shape[1]=-1
    w=np.reshape(w,shape)
    return np.clip(x,0,np.inf)+w*np.clip(x,-np.inf,0)


def pool(x, pool_size, strides, padding, pool_mode):

    if padding == 'same':
        pad = [(0, 0), (0, 0)] + [(s // 2, s // 2) for s in pool_size]
        x = np.pad(x, pad, 'constant', constant_values=-np.inf)

    # indexing trick
    x = np.pad(x, [(0, 0), (0, 0)] + [(0, 1) for _ in pool_size],
               'constant', constant_values=0)

    y = []
    for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0)):
        for (l, l1) in zip(range(pool_size[1]), range(-pool_size[1], 0)):
            y.append(x[:, :, k:k1:strides[0], l:l1:strides[1]])

    y = np.stack(y, axis=-1)
    if pool_mode == 'avg':
        y = np.mean(np.ma.masked_invalid(y), axis=-1).data
    elif pool_mode == 'max':
        y = np.max(y, axis=-1)

    return y

def maxpool(x,pool_size=(2,2),strides=(2,2),padding='same'):
    return pool(x,pool_size,strides,padding,'max')

def fc(x,w,bias=None):
    x=np.reshape(x,(x.shape[0],-1))
    out=np.matmul(x,np.transpose(w))
    if bias is not None:
        bias = np.reshape(bias, (1,-1))
        return out+bias
    return out


def softmax(target, axis):
    max_axis = np.max(target, axis, keepdims=True)
    target_exp = np.exp(target - max_axis)
    normalize = np.sum(target_exp, axis, keepdims=True)
    softmax =target_exp/normalize
    return softmax
def sigmoid(x):
    return 1/(1+np.exp(-1*x))


def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12

    imap = imap
    dx1 = reg[0,:,:]
    dy1 = reg[1,:,:]
    dx2 = reg[2,:,:]
    dy2 = reg[3,:,:]
    y, x = np.where(imap >= t)
    # if y.shape[0] == 1:
    #     dx1 = np.flipud(dx1)
    #     dy1 = np.flipud(dy1)
    #     dx2 = np.flipud(dx2)
    #     dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]])
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.vstack([x,y])
    q1 = np.fix((stride * bb + 1) )
    q2 = np.fix((stride * bb + cellsize - 1 + 1))
    if len(bb.shape)==1 or bb.shape[1]<1:
        return [],[]
    else:
        w = q2[0,:] - q1[0,:] + 1
        h = q2[1,:] - q1[1,:] + 1
        b1 = np.round(np.expand_dims(q1[0,:]+ reg[0,:] * w,-1))/scale
        b2 = np.round(np.expand_dims(q1[1,:]+ reg[ 1,:] * h,-1))/scale
        b3 = np.round(np.expand_dims(q2[0,:] +reg[2,:] * w,-1))/scale
        b4 = np.round(np.expand_dims(q2[1,:] + reg[3,:] * h,-1))/scale

        boundingbox= np.concatenate([b1, b2, b3, b4, np.expand_dims(score, -1)],axis=-1)
        return boundingbox, reg

def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        overlap1 = inter / area[i]
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold )[0])))
        if len(idxs>0):
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap1 > overlap_threshold+0.2)[0])))
    return boxes[pick]

def nms_numpy(bboxes, nms_thresh=0.5):
    """
    bboxes: num_insts x 5 [batch,anchor ,x1,y1,x2,y2,conf,class,class conf]
    """
    if len(bboxes)==0:
        return bboxes

    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    conf = bboxes[:,4]

    area_all = (x2-x1)*(y2-y1)
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []

    while len(sorted_index)>0:
        # get the last biggest values
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        # pop the value
        sorted_index = sorted_index[:-1]
        # get the remaining boxes
        yy1 = np.take(y1, indices=sorted_index)
        xx1 = np.take(x1, indices=sorted_index)
        yy2 = np.take(y2, indices=sorted_index)
        xx2 = np.take(x2, indices=sorted_index)
        xc=np.abs((xx1+xx2)/2-( x1[curr_index]+ x2[curr_index])/2)
        yc=np.abs((yy1+yy2)/2-( y1[curr_index]+ y2[curr_index])/2)

        # get the intersection box
        yy1 = np.maximum(yy1, y1[curr_index])
        xx1 = np.maximum(xx1, x1[curr_index])
        yy2 = np.minimum(yy2, y2[curr_index])
        xx2 = np.minimum(xx2, x2[curr_index])
        # calculate IoU
        w = xx2-xx1
        h = yy2-yy1

        w = np.maximum(0., w)
        h = np.maximum(0., h)
        xc_diff = np.less(xc - 0.4 * w,0).astype(np.float32)
        yc_diff = np.less(yc - 0.4 * h,0).astype(np.float32)
        close_idx=xc_diff+yc_diff
        inter = w*h
        rem_areas = np.take(area_all, indices=sorted_index)
        union = (rem_areas-inter)+area_all[curr_index]
        IoU = inter/union
        IoU1 = inter / rem_areas
        IoU2 = inter /area_all[curr_index]
        IoU[IoU1>0.9]=0.9
        IoU[IoU2 > 0.9] =0.9
        IoU[close_idx==2]=0.9
        sorted_index = sorted_index[IoU<=nms_thresh ]

    out_bboxes = np.take(bboxes, keep_index, axis=0)

    return out_bboxes

def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph



# img=cv2.imread('kim48.jpg')[:,8:56,:]
# img48=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).transpose([2,0,1])
# img24=cv2.cvtColor(cv2.resize(img,(24,24),cv2.INTER_AREA),cv2.COLOR_BGR2RGB).transpose([2,0,1])
# img12=cv2.cvtColor(cv2.resize(img,(12,12),cv2.INTER_AREA),cv2.COLOR_BGR2RGB).transpose([2,0,1])
# img48 = np.expand_dims((img48 - 127.5) * 0.0078125,0)
# img24 = np.expand_dims((img24 - 127.5) * 0.0078125,0)
# img12 = np.expand_dims((img12 - 127.5) * 0.0078125,0)
#
#


#輸入為12x12
def P_net(inp):
    try:
        out=conv(inp,det1['pre_layer.0.weight'],det1['pre_layer.0.bias'],strides=(1,1),padding='valid')
        out=prelu(out,det1['pre_layer.1.weight'])
        out=maxpool(out,(2,2),(2,2),padding='valid')
        out=conv(out,det1['pre_layer.3.weight'],det1['pre_layer.3.bias'],strides=(1,1),padding='valid')
        out=prelu(out,det1['pre_layer.4.weight'])
        out=conv(out,det1['pre_layer.5.weight'],det1['pre_layer.5.bias'],strides=(1,1),padding='valid')
        out=prelu(out,det1['pre_layer.6.weight'])
        out1=conv(out,det1['conv4_1.weight'],det1['conv4_1.bias'],strides=(1,1),padding='valid')
        out1=sigmoid(out1)
        out2=conv(out,det1['conv4_2.weight'],det1['conv4_2.bias'],strides=(1,1),padding='valid')
        out3 = conv(out, det1['conv4_3.weight'], det1['conv4_3.bias'], strides=(1, 1), padding='valid')
        return out1, out2,out3
    except Exception as e:
        print(inp.shape)
        print(e)

    return None,None
#輸入為24x24



def R_net(inp):
    out=conv(inp,det2['pre_layer.0.weight'],det2['pre_layer.0.bias'],strides=(1,1),padding='valid')
    out=prelu(out,det2['pre_layer.1.weight'])
    out=maxpool(out,(3,3),(2,2),padding='valid')
    out=conv(out,det2['pre_layer.3.weight'],det2['pre_layer.3.bias'],strides=(1,1),padding='valid')
    out=prelu(out,det2['pre_layer.4.weight'])
    out=maxpool(out,(3,3),(2,2),padding='valid')
    out=conv(out,det2['pre_layer.6.weight'],det2['pre_layer.6.bias'],strides=(1,1),padding='valid')
    out=prelu(out,det2['pre_layer.7.weight'])
    out=fc(out,det2['conv4.weight'],det2['conv4.bias'])
    out=prelu(out,det2['prelu4.weight'])
    out1=fc(out,det2['conv5_1.weight'],det2['conv5_1.bias'])
    out1 = sigmoid(out1)
    out2=fc(out,det2['conv5_2.weight'],det2['conv5_2.bias'])
    out3 = fc(out, det2['conv5_3.weight'], det2['conv5_3.bias'])
    return out1,out2,out3

#48x48
def O_net(inp):
    out = conv(inp, det3['pre_layer.0.weight'],det3['pre_layer.0.bias'], strides=(1, 1), padding='valid')
    out = prelu(out, det3['pre_layer.1.weight'])

    out = maxpool(out, (3, 3), (2, 2),padding='valid')
    out = conv(out, det3['pre_layer.3.weight'],det3['pre_layer.3.bias'], strides=(1, 1), padding='valid')

    out = prelu(out, det3['pre_layer.4.weight'])

    out = maxpool(out, (3, 3), (2, 2),padding='valid')

    out = conv(out, det3['pre_layer.6.weight'],det3['pre_layer.6.bias'], strides=(1, 1), padding='valid')
    out = prelu(out, det3['pre_layer.7.weight'])
    out = maxpool(out, (2, 2), (2, 2), padding='valid')

    out = conv(out, det3['pre_layer.9.weight'],det3['pre_layer.9.bias'], strides=(1, 1), padding='valid')

    out = prelu(out, det3['pre_layer.10.weight'])

    out = fc(out, det3['conv5.weight'],det3['conv5.bias'])
    out = prelu(out, det3['prelu5.weight'])

    out1 = fc(out, det3['conv6_1.weight'], det3['conv6_1.bias'])
    out1 = sigmoid(out1)

    out2 = fc(out, det3['conv6_2.weight'], det3['conv6_2.bias'])
    out3 = fc(out, det3['conv6_3.weight'], det3['conv6_3.bias'])
    return out1, out2,out3


image_name='kim'
#minsize=20, threshold=0.7, factor=0.709, use_auto_downscaling=True,min_face_area=25 * 25)
factor_count=0
minsize=5
threshold= [0.6, 0.7, 0.7]
factor=0.709
#img=Image.open(image_name+'.jpg')
img=cv2.imread(image_name+'.jpg')
#img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
min_face_area=(minsize ,minsize)
h = img.shape[0]
w = img.shape[1]
minl = np.amin([h, w])
m = 12.0 / minsize
minl = minl * m
# create scale pyramid
scales = []
while minl >= 12:
    scales += [m * np.power(factor, factor_count)]
    minl = minl * factor
    factor_count += 1
all_boxes=[]
for scale in scales:
    hs = int(np.ceil(h * scale))
    ws = int(np.ceil(w * scale))

    #=img_data=np.array(img.resize((ws, hs), Image.BILINEAR)).transpose((2, 0, 1))
    #img_data = img_data/255.
    img_data=cv2.resize(img,(ws,hs)).transpose((2, 0, 1))/255.0
    img_data = np.expand_dims(img_data, 0)
    p_out1,p_out2,p_out3 = P_net(img_data)
    if p_out1 is not None:
        boxes, _ = generateBoundingBox(p_out1[0,0,:,:].copy(), p_out2[0,:,:,:].copy(), scale, threshold[0])
        boxes=nms_numpy(boxes,0.5)

        if len(boxes) > 0:
            im1=img.copy()
            boxes = rerec(boxes)
            accept=0
            accept_score=0
            for box in boxes:
                sub_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]


                if sub_img is not None and len(sub_img.shape)==3 and sub_img.shape[1]>=min_face_area[0] and sub_img.shape[0]>=min_face_area[1]:
                    #extend_len=int(round(min([int(box[1]),int(box[0]),img.shape[0]-int(box[3]),img.shape[1]-int(box[2]),0.5*boxw])))
                    cv2.imwrite('faces/sub_img_{0}_{1}_{2}_{3}.jpg'.format(int(box[0]),int(box[1]),int(box[2]),int(box[3])),cv2.cvtColor(sub_img.copy(), cv2.COLOR_RGB2BGR))
                    sub_img =cv2.resize(sub_img, (24, 24)).transpose((2, 0, 1)) / 255.0
                    sub_img = np.expand_dims(sub_img, 0)
                    r_out1, r_out2,r_out3 = R_net(sub_img)
                    avg_score=math.sqrt(r_out1[0,0]*box[4])
                    if r_out1[0,0]>=threshold[1]:
                        accept +=1
                        accept_score +=avg_score
                        x1 = np.round(box[0]+r_out2[0,0]*sub_img.shape[1])
                        y1 = np.round(box[1]+r_out2[0,1]*sub_img.shape[0])
                        x2 =  np.round(box[2]+r_out2[0,2]*sub_img.shape[1])
                        y2 = np.round(box[3]+r_out2[0,3]*sub_img.shape[0])
                        all_boxes.append(np.array([int(x1),int(y1),int(x2),int(y2),avg_score,r_out3[0]]))
                        cv2.rectangle(im1,(int(x1),int(y1)),(int(x2),int(y2)), (255, 128, 255), 2)

            cv2.imwrite('box_{0}_{1}.jpg'.format(image_name,scale),cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))
            print('scale:{0} finished!   {1}  box founded  avg score:{2} '.format(scale,len(boxes),boxes[:,4].mean()))
            print( '          {0}  box accepted  avg score:{1} '.format(accept, accept_score/max(accept,1)))
            accept=0
            accept_score=0
        else:
            print('scale:{0} finished!   {1}  box founded   '.format(scale, len(boxes)))
print('{0} boxes accepted'.format(len(all_boxes)))
all_boxes=nms_numpy(np.asarray(all_boxes),0.5)
print('{0} boxes accepted after nms'.format(len(all_boxes)))
im2=img.copy()
for b in all_boxes:
    x1, y1, x2, y2, score, landmarks = b
    landmarks_x = x1 + landmarks[0::2] * (x2 - x1)
    landmarks_y = y1 + landmarks[1::2] * (y2 - y1)
    for i in range(5):
        cv2.circle(im2, (int(landmarks_x[i]), int(landmarks_y[i])), 2, (128, 255, 255), 1)
    cv2.rectangle(im2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 128), 2)
cv2.imwrite('box_'+image_name+'_rnet.jpg',cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
all_boxes=np.asarray(all_boxes)
all_boxes=rerec(all_boxes)
im3=img.copy()
final_boxes=[]
for box in all_boxes:
    boxw = box[2] - box[0]
    boxh = box[3] - box[1]
    sub_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    if sub_img is not None and sub_img.shape[0]>0 and sub_img.shape[1]>0 :
        cv2.imwrite('faces/final/sub_img_{0}_{1}_{2}_{3}.jpg'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3])),cv2.cvtColor(sub_img.copy(), cv2.COLOR_RGB2BGR))
        sub_img = cv2.resize(sub_img, (48, 48)).transpose((2, 0, 1)) / 255.0
        sub_img = np.expand_dims(sub_img, 0)
        o_out1, o_out2,o_out3 = O_net(sub_img)
        avg_score = o_out1[0, 0]
        if avg_score>=threshold[2]:
            x1 = np.round(box[0] + o_out2[0, 0] * boxw)
            y1 = np.round(box[1] + o_out2[0, 1] * boxh)
            x2 = np.round(box[2] + o_out2[0, 2] * boxw)
            y2 = np.round(box[3] + o_out2[0, 3] * boxh)
            final_boxes.append(np.array([int(x1),int(y1), int(x2), int(y2), avg_score,o_out3[0]]))


final_boxes=nms_numpy(np.asarray(final_boxes),0.5)
for b in final_boxes:
    x1,y1,x2,y2,score,landmarks=b

    landmarks_x=x1+landmarks[0::2]*(x2-x1)
    landmarks_y=y1+landmarks[1::2]*(y2-y1)
    for i in range(5):
        cv2.circle(im3, (int(landmarks_x[i]), int(landmarks_y[i])),2, (128, 255, 255),1)
    cv2.rectangle(im3, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 128), 2)
cv2.imwrite('box_'+image_name+'_onet.jpg',cv2.cvtColor(im3, cv2.COLOR_RGB2BGR))
print(final_boxes)
















#
# p_out1,p_out2=P_net(img12)
# print('p_out1: {0} p_out2:{1}'.format(p_out1.shape,p_out2.shape))
#
# bbox=generateBoundingBox(p_out1[0,1].copy(),p_out2[0,:].copy(),256/12)
#
#
# r_out1,r_out2=R_net(img24)
# print('r_out1: {0} r_out2:{1}'.format(r_out1.shape,r_out2.shape))
# o_out1,o_out2=O_net(img48)
# print('o_out1: {0} o_out2:{1}'.format(o_out1.shape,o_out2.shape))
#
#
#

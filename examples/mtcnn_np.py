import numpy as np
import scipy.signal as signal
import scipy as sp
import cv2
import math
det1=np.load('det1.npy',allow_pickle=True, encoding='latin1').item()
det2=np.load('det2.npy',allow_pickle=True, encoding='latin1').item()
det3=np.load('det3.npy',allow_pickle=True, encoding='latin1').item()


def normalize_conv(func):
    def wrapper(*args, **kwargs):
        x = args[0]
        w = args[1]

        w = np.fliplr(np.flipud(w))
        w = np.transpose(w, (2, 3, 0, 1))
        dilation_rate = kwargs.pop('dilation_rate', 1)
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate,) * (x.ndim - 2)
        if dilation_rate!=(1,1):
            for (i, d) in enumerate(dilation_rate):
                if d > 1:
                    for j in range(w.shape[2 + i] - 1):
                        w = np.insert(w, 2 * j + 1, 0, axis=2 + i)

        y = func(x, w, **kwargs)
        # if kwargs['data_format'] == 'channels_last':
        #     if y.ndim == 3:
        #         y = np.transpose(y, (0, 2, 1))
        #     elif y.ndim == 4:
        #         y = np.transpose(y, (0, 2, 3, 1))
        #     else:
        #         y = np.transpose(y, (0, 2, 3, 4, 1))

        return y

    return wrapper


@normalize_conv
def conv(x, w, strides,padding):
    y = []
    for i in range(x.shape[0]):
        _y = []
        for j in range(w.shape[1]):
            __y = []
            for k in range(w.shape[0]):
                __y.append(signal.convolve(x[i, k], w[k, j], mode=padding)[::strides[0], ::strides[1]])
            _y.append(np.sum(np.stack(__y, axis=-1), axis=-1))
        y.append(_y)
    y = np.array(y)
    return y

@normalize_conv
def conv1(x, w, strides,padding):
    y = []
    for i in range(x.shape[0]):
        _y = []
        for j in range(w.shape[1]):
            __y = []
            for k in range(w.shape[0]):
                __y.append(np.convolve(x[i, k], w[k, j], mode=padding)[::strides[0], ::strides[1]])
            _y.append(np.sum(np.stack(__y, axis=-1), axis=-1))
        y.append(_y)
    y = np.array(y)
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
    out=np.matmul(x,w)
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap1 > overlap_threshold+0.2)[0])))
    return boxes[pick]


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
        out=conv(inp,det1['conv1']['weights'],strides=(1,1),padding='valid')

        out=out+np.reshape(det1['conv1']['biases'],(1,-1,1,1))
        out=prelu(out,det1['PReLU1']['alpha'])

        out=maxpool(out,(2,2),(2,2),padding='valid')

        out=conv(out,det1['conv2']['weights'],strides=(1,1),padding='valid')

        out=out+np.reshape(det1['conv2']['biases'],(1,-1,1,1))
        out=prelu(out,det1['PReLU2']['alpha'])

        out=conv(out,det1['conv3']['weights'],strides=(1,1),padding='valid')

        out=out+np.reshape(det1['conv3']['biases'],(1,-1,1,1))
        out=prelu(out,det1['PReLU3']['alpha'])

        out1=conv(out,det1['conv4-1']['weights'],strides=(1,1),padding='valid')

        out1=out1+np.reshape(det1['conv4-1']['biases'],(1,-1,1,1))
        out1=softmax(out1,1)

        out2=conv(out,det1['conv4-2']['weights'],strides=(1,1),padding='valid')

        out2=out2+np.reshape(det1['conv4-2']['biases'],(1,-1,1,1))
        return out1, out2
    except Exception as e:
        print(inp.shape)
        print(e)

    return None,None
#輸入為24x24
def R_net(inp):
    out=conv(inp,det2['conv1']['weights'],strides=(1,1),padding='valid')

    out=out+np.reshape(det2['conv1']['biases'],(1,-1,1,1))
    out=prelu(out,det2['prelu1']['alpha'])

    out=maxpool(out,(3,3),(2,2))
    out=conv(out,det2['conv2']['weights'],strides=(1,1),padding='valid')

    out=out+np.reshape(det2['conv2']['biases'],(1,-1,1,1))
    out=prelu(out,det2['prelu2']['alpha'])

    out=maxpool(out,(3,3),(2,2),padding='valid')

    out=conv(out,det2['conv3']['weights'],strides=(1,1),padding='valid')

    out=out+np.reshape(det2['conv3']['biases'],(1,-1,1,1))
    out=prelu(out,det2['prelu3']['alpha'])

    out=fc(out,det2['conv4']['weights'],det2['conv4']['biases'])
    out=prelu(out,det2['prelu4']['alpha'])

    out1=fc(out,det2['conv5-1']['weights'],det2['conv5-1']['biases'])
    out1 = softmax(out1,1)

    out2=fc(out,det2['conv5-2']['weights'],det2['conv5-2']['biases'])
    return out1,out2

#48x48
def O_net(inp):
    out = conv(inp, det3['conv1']['weights'], strides=(1, 1), padding='valid')

    out = out + np.reshape(det3['conv1']['biases'], (1, -1, 1, 1))
    out = prelu(out, det3['prelu1']['alpha'])

    out = maxpool(out, (3, 3), (2, 2))
    out = conv(out, det3['conv2']['weights'], strides=(1, 1), padding='valid')

    out = out + np.reshape(det3['conv2']['biases'], (1, -1, 1, 1))
    out = prelu(out, det3['prelu2']['alpha'])

    out = maxpool(out, (3, 3), (2, 2),padding='valid')

    out = conv(out, det3['conv3']['weights'], strides=(1, 1), padding='valid')

    out = out + np.reshape(det3['conv3']['biases'], (1, -1, 1, 1))
    out = prelu(out, det3['prelu3']['alpha'])
    out = maxpool(out, (2, 2), (2, 2), padding='valid')

    out = conv(out, det3['conv4']['weights'], strides=(1, 1), padding='valid')
    out = out + np.reshape(det3['conv4']['biases'], (1, -1, 1, 1))
    out = prelu(out, det3['prelu4']['alpha'])

    out = fc(out, det3['conv5']['weights'], det3['conv5']['biases'])
    out = prelu(out, det3['prelu5']['alpha'])

    out1 = fc(out, det3['conv6-1']['weights'], det3['conv6-1']['biases'])
    out1 = softmax(out1,1)

    out2 = fc(out, det3['conv6-2']['weights'], det3['conv6-2']['biases'])
    return out1, out2


image_name='JutisfyMyLove'
#minsize=20, threshold=0.7, factor=0.709, use_auto_downscaling=True,min_face_area=25 * 25)
factor_count=0
minsize=20
threshold= [0.7, 0.7, 0.9]
factor=0.709
img=cv2.imread(image_name+'.jpg')
min_face_area=(25 ,25)
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
    img_data= cv2.resize(img.copy(), (ws,hs),cv2.INTER_AREA)
    img_data = (img_data - 127.5) * 0.0078125
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    p_out1,p_out2 = P_net(img_data)
    if p_out1 is not None:
        boxes, _ = generateBoundingBox(p_out1[0,1,:,:].copy(), p_out2[0,:,:,:].copy(), scale, threshold[0])
        boxes=nms(boxes,0.5)
        if len(boxes) > 0:
            im1=img.copy()
            boxes=rerec(boxes)
            accept=0
            accept_score=0
            for box in boxes:
                boxw=box[2]-box[0]
                boxh=box[3] -box[1]
                if boxw>=min_face_area[0] and boxh>=min_face_area[1]:
                    #extend_len=int(round(min([int(box[1]),int(box[0]),img.shape[0]-int(box[3]),img.shape[1]-int(box[2]),0.5*boxw])))
                    sub_img=img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
                    cv2.imwrite('faces/sub_img_{0}_{1}_{2}_{3}.jpg'.format(int(box[0]),int(box[1]),int(box[2]),int(box[3])),sub_img)
                    sub_img = cv2.resize(sub_img, (24, 24), cv2.INTER_AREA)
                    sub_img = (sub_img - 127.5) * 0.0078125
                    sub_img = np.transpose(sub_img, (2, 0, 1))
                    sub_img = np.expand_dims(sub_img, 0)
                    r_out1, r_out2 = R_net(sub_img)
                    avg_score=math.sqrt(r_out1[0,1]*box[4])
                    if r_out1[0,1]>=threshold[1]:
                        accept +=1
                        accept_score +=avg_score
                        x1 = np.round(box[0]+r_out2[0,0]*boxw)
                        y1 = np.round(box[1]+r_out2[0,1]*boxh)
                        x2 =  np.round(box[2]+r_out2[0,2]*boxw)
                        y2 = np.round(box[3]+r_out2[0,3]*boxh)
                        all_boxes.append(np.array([int(x1),int(y1),int(x2),int(y2),avg_score]))
                        cv2.rectangle(im1,(int(x1),int(y1)),(int(x2),int(y2)), (255, 128, 255), 3)

            cv2.imwrite('box_{0}_{1}.jpg'.format(image_name,scale),im1)
            print('scale:{0} finished!   {1}  box founded  avg score:{2} '.format(scale,len(boxes),boxes[:,4].mean()))
            print( '          {0}  box accepted  avg score:{1} '.format(accept, accept_score/max(accept,1)))
            accept=0
            accept_score=0
        else:
            print('scale:{0} finished!   {1}  box founded   '.format(scale, len(boxes)))
print('{0} boxes accepted'.format(len(all_boxes)))
all_boxes=nms(np.asarray(all_boxes),0.5)
print('{0} boxes accepted after nms'.format(len(all_boxes)))
im2=img.copy()
for b in all_boxes:
    cv2.rectangle(im2, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (128, 128, 255), 3)
cv2.imwrite('box_'+image_name+'_rnet.jpg',im2)
all_boxes=np.asarray(all_boxes)
all_boxes=rerec(all_boxes)
im3=img.copy()
final_boxes=[]
for box in all_boxes:
    boxw = box[2] - box[0]
    boxh = box[3] - box[1]
    sub_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    if sub_img is not None and sub_img.shape[0]>0 and sub_img.shape[1]>0 :
        cv2.imwrite('faces/final/sub_img_{0}_{1}_{2}_{3}.jpg'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3])),sub_img)
        sub_img = cv2.resize(sub_img, (48, 48), cv2.INTER_AREA)
        sub_img = (sub_img - 127.5) * 0.0078125
        sub_img = np.transpose(sub_img, (2, 0, 1))
        sub_img = np.expand_dims(sub_img, 0)
        o_out1, o_out2 = O_net(sub_img)
        avg_score = o_out1[0, 1]
        if avg_score>=threshold[2]:
            x1 = np.round(box[0] + o_out2[0, 0] * boxw)
            y1 = np.round(box[1] + o_out2[0, 1] * boxh)
            x2 = np.round(box[2] + o_out2[0, 2] * boxw)
            y2 = np.round(box[3] + o_out2[0, 3] * boxh)
            final_boxes.append(np.array([int(x1),int(y1), int(x2), int(y2), avg_score]))
            cv2.rectangle(im3, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 128), 3)
cv2.imwrite('box_'+image_name+'_onet.jpg',im3)
final_boxes=nms(np.asarray(final_boxes),0.5)
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

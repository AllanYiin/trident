import os
import random
import re
import numpy as np
from .load_backend import *
from .common import image_data_format


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def add_noise(image):
    rr=random.randint(0,10)
    noise = np.random.standard_normal(image.shape) * np.random.choice(np.arange(5,15))
    if rr%2==0:
        noise = np.random.uniform(-1,1,image.shape) * np.random.choice(np.arange(5,15))
    image=np.clip(image+noise,0,255)
    return image




def random_channel_shift(x, intensity=15., channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    intensity = max_x / 255 * intensity
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_cutout(img, mask ):
    h, w = img.shape[:2] if image_data_format()=='channels_last' or len(img.shape)==2 else img.shape[1:3]
    cutx=random.choice(range(0,w//4))
    cuty = random.choice(range(0, h//4))
    offsetx=random.choice(range(0,w))
    offsety=random.choice(range(0,h))
    block=np.zeros((min(offsety+cuty,h)-offsety,min(offsetx+cutx,w)-offsetx))
    if random.randint(0, 10) % 4 == 1:
        block = np.clip(np.random.standard_normal((min(offsety+cuty,h)-offsety,min(offsetx+cutx,w)-offsetx)) * 127.5 + 127.5, 0,255)
    elif random.randint(0, 10) % 4 == 2:
        block = np.random.uniform(0,255,(min(offsety + cuty, h) - offsety, min(offsetx + cutx, w) - offsetx))
    if image_data_format() == 'channels_last':
        block=np.expand_dims(block,-1)
        block = np.concatenate([block, block, block], axis=-1)
        img[offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1]), :] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    else:
        block = np.expand_dims(0,block)
        block = np.concatenate([block, block, block], axis=0)
        img[:,offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1])] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    return img,mask




def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]
    return image, mask

def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)

    return image, mask

def do_random_crop_rotate_rescale(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1,0.1,2)

    dscale_x = np.random.uniform(-0.00075,0.00075)
    dscale_y = np.random.uniform(-0.25,0.25)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift*min(H,W)

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+W/2
    y = (src*[sin, cos]).sum(1)+H/2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = cv2.warpPerspective( mask, transform, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    return image, mask

def do_random_log_contast(image, gain=[0.70, 1.30] ):
    gain = np.random.uniform(gain[0],gain[1],1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image


def random_contast(image):
    beta=0
    alpha=random.uniform(0.5, 2.0)
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image,0,255).astype(np.uint8)
    return image










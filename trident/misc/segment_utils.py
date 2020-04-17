import cv2
import numpy as np
__all__ = ['generate_random_trimap']

def generate_random_trimap(mask):
    mask=mask.copy().astype(np.float32)
    mask[mask>0]=255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if (mask.shape[0] * mask.shape[1] < 250000):
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(5, 7))
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(7, 10))
    else:
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(10, 15))
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(15, 20))

    ### 操作矩阵生成trimap，特别快
    # ((mask-erode)==255.0)*128  腐蚀掉的区域置为128
    # ((dilate-mask)==255.0)*128 膨胀出的区域置为128
    # + erode 整张图变为255/0/128
    img_trimap = ((mask - erode) == 255.0) * 128 + ((dilate - mask) == 255.0) * 128 + erode

    return img_trimap.astype(np.uint8)
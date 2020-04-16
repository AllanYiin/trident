

def generate_random_trimap(alpha):
    ### 非0区域置为255，然后膨胀及收缩，多出的部分为128区域
    ### 优点：如果有一小撮头发为小于255，但大于0的，那通过该方法，128区域会覆盖到该一小撮头发部分
    mask = alpha.copy()  # 0~255
    # 非纯背景置为255
    mask = ((mask != 0) * 255).astype(np.float32)  # 0.0和255.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 如果尺寸过小(总面积小于500*500)，则减半膨胀和腐蚀的程度
    if (alpha.shape[0] * alpha.shape[1] < 250000):
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(5, 7))  # 膨胀少点
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(7, 10))  # 腐蚀多点
    else:
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(10, 15))  # 膨胀少点
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(15, 20))  # 腐蚀多点

    ### for循环生成trimap，特别慢
    # for row in range(mask.shape[0]):
    #     for col in range(mask.shape[1]):
    #         # 背景区域为第0类
    #         if(dilate[row,col]==255 and mask[row,col]==0):
    #             img_trimap[row,col]=128
    #         # 前景区域为第1类
    #         if(mask[row,col]==255 and erode[row,col]==0):
    #             img_trimap[row,col]=128

    ### 操作矩阵生成trimap，特别快
    # ((mask-erode)==255.0)*128  腐蚀掉的区域置为128
    # ((dilate-mask)==255.0)*128 膨胀出的区域置为128
    # + erode 整张图变为255/0/128
    img_trimap = ((mask - erode) == 255.0) * 128 + ((dilate - mask) == 255.0) * 128 + erode

    return img_trimap.astype(np.uint8)
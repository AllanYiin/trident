import dlib
import numpy as np


__all__ = ['get_faces','crop_faces']



detector = dlib.get_frontal_face_detector()
landmark_predictor =None# dlib.shape_predictor('../Data/ex07_train/shape_predictor_68_face_landmarks.dat')



def get_faces(img):
    face_rects = detector(img, 1)
    faces=[]
    for i, d in enumerate(face_rects):
        # 讀取框左上右下座標
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        faces.append([x1,y1,x2,y2])
    return faces

def crop_faces(img):
    face_rects = detector(img, 1)
    faces=[]
    for i, d in enumerate(face_rects):
        # 讀取框左上右下座標
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        faces.append(img[y1-1:y2,x1-1:x2,:])
    return faces



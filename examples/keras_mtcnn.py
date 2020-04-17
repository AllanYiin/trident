from mtcnn.mtcnn import MTCNN
import cv2

img = cv2.cvtColor(cv2.imread("kim.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
detector.detect_faces(img)
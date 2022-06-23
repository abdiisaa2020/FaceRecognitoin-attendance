from types import GeneratorType
import cv2
import os
import numpy as np
import cv2

def faceDetection(test_img):
    gray_img= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('C:\\Users\\zsuperbin\\Desktop\\frs\\haarcascade_frontalface_default.xml')
    # ?face_haar_cascade=cv2.CascadeClassifier('C:\\Users\\zsuperbin\\Desktop\\frs\\haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)

    return faces, gray_img
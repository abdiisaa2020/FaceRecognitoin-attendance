import cv2
import numpy as np
import os
import faceRecognition as fr
test_img= cv2.imread('C:\\Users\\zsuperbin\\Desktop\\frs\\images\\abdi.jpg')
faces_detected,gray_img =fr.faceDetection(test_img)
print("face_detected:", faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=25)
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face_detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

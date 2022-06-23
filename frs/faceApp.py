from turtle import width
from unittest import result
import cv2
import numpy as np
import face_recognition as face_rec


# resizing i\the image
def resize(img, size):
    width= int(img.shape[1]*size)
    height=int(img.shape[0]*size)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)


abdi= face_rec.load_image_file('C:\\Users\\zsuperbin\\Desktop\\frs\\images\\abdi.jpg')
abdi=cv2.cvtColor(abdi, cv2.COLOR_BGR2RGB)
abdi= resize(abdi,0.50)

abditest= face_rec.load_image_file('C:\\Users\\zsuperbin\\Desktop\\frs\\images\\sb.jpg')
abditest= cv2.cvtColor(abditest,cv2.COLOR_BGR2RGB)
abditest=resize(abditest,0.50)

# finding face location
faceLocation_abdi=face_rec.face_locations(abdi)[0]
encode_abdi=face_rec.face_encodings(abdi)[0]
cv2.rectangle(abdi,(faceLocation_abdi[3],faceLocation_abdi[0]),(faceLocation_abdi[1],faceLocation_abdi[2]),(255,0,255),5)

faceLocation_abditest=face_rec.face_locations(abditest)[0]
encode_abditest=face_rec.face_encodings(abditest)[0]
cv2.rectangle(abditest,(faceLocation_abditest[3],faceLocation_abditest[0]),(faceLocation_abditest[1],faceLocation_abditest[2]),(255,0,255),5)

result = face_rec.compare_faces([encode_abdi],encode_abditest)
print(result)
cv2.putText(abditest,f'{result}',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)

cv2.imshow("main_image", abdi)
cv2.imshow("test_image",abditest)
cv2.waitKey(0)
cv2.destroyAllWindows



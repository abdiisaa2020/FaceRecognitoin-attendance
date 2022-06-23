from datetime import datetime
from turtle import width
from xml.etree.ElementPath import find
import cv2
from cv2 import VideoCapture
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeech

engine = textSpeech.init()

def resize(img, size):
    width =int(img.shape[1]*size)
    height=int(img.shape[0]*size)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)


path= 'student_images'
studentimg= []
studentName=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    currentImage=cv2.imread(f'{path}\{cl}')
    studentimg.append(currentImage)
    studentName.append(os.path.splitext(cl)[0])
# print(studentName)

def findEncoding(images):
    encodingList=[]
    for img in images:
        img= resize(img,0.50)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg= face_rec.face_encodings(img)[0]
        encodingList.append(encodeimg)
    return encodingList

#Marking attendance
def MarkAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList =f.readlines()
        nameList =[]

        for line in myDataList:
            entry = line.split(',')
            nameList.append(encodingList[0])

        if name not in nameList:
            now=datetime.now()
            timestr= now.strftime('%H:%M') 
            f.writelines(f'\n{name},{timestr}') 
            engine.say(' welcome to class'+name) 
            engine.runAndWait()

encodingList =findEncoding(studentimg)
vid=cv2.VideoCapture(0)

while True:
    success,frame= vid.read()
    smaller_frames= cv2.resize(frame,(0,0),None,0.25,0.25)
    smaller_frames=cv2.cvtColor(smaller_frames,cv2.COLOR_BGR2RGB)

    faces_in_frame=face_rec.face_locations(smaller_frames)
    encode_in_frame=face_rec.face_encodings(smaller_frames,faces_in_frame)

    for encodeFace,faceloc in zip(encode_in_frame,faces_in_frame):
        matches=face_rec.compare_faces(encodingList,encodeFace)
        faceDis=face_rec.face_distance(encodingList,encodeFace)
        print(faceDis)
        matchIndex= np.argmin(faceDis)

        if matches[matchIndex]:
            name= studentName[matchIndex]
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MarkAttendance(name)
        cv2.imshow('video',frame)
        cv2.waitKey(1)












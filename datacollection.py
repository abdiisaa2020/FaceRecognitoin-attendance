import cv2
import os
video= cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier("C:\\Users\\zsuperbin\\Desktop\\FaceRecognitoin-attendance\\haarcascade_frontalface_default.xml")
count =0
nameID= str(input("Enter your name:")).lower()

path='images/'+nameID
isExist=os.path.exists(path)

if isExist:
    print("name already taken")
    nameID  =str(input("Enter your name Agian"))
else:
    os.makedirs(path)

while True:
    ret,frame = video.read()
    faces= facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        count= count+1
        name='./images/'+nameID+'/'+str(count)+'.jpg'
        print("Creating Images ................."+name)
        cv2.imwrite(name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y),(x+w,y+h),(233,255,0),5)
        
    cv2.imshow("window frame",frame)
    cv2.waitKey(1)
    if count>300:
        break
video.release()
cv2.destroyAllWindows





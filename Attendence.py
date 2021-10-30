import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
def findEnconde(images : list):
    encondelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encondelist.append(encodeImg)
    return encondelist

def Markattendence(name : str):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',') ##name is first element
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

##we want to take evrey pic in the images lib and enconde it
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)


for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

encodelistKnown = findEnconde(images)
##enconding Complete


cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img , (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img , (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
            Markattendence(str(name))


    cv2.imshow('Webcam' , img)
    if cv2.waitKey(20) & 0XFF == ord('q'):
        print("Done. Thanks! :-)")
        break

cap.release()
cv2.destroyAllWindows()








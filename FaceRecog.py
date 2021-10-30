import cv2
import numpy as np
import face_recognition


##uploading the image
imgElon = face_recognition.load_image_file('Images/elon_musk_face.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/elon_musk_side.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgBillTest = face_recognition.load_image_file('Images/bill_gates_face.jpg')
imgBillTest = cv2.cvtColor(imgBillTest,cv2.COLOR_BGR2RGB)

##face location at the image
faceLoc = face_recognition.face_locations(imgElon)[0] ##first elemnt in here
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0] ##first elemnt in here
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

faceLocBillTest = face_recognition.face_locations(imgBillTest)[0] ##first elemnt in here
encodeBillTest = face_recognition.face_encodings(imgBillTest)[0]
cv2.rectangle(imgBillTest,(faceLocBillTest[3],faceLocBillTest[0]),(faceLocBillTest[1],faceLocBillTest[2]),(255,0,255),2)

##comapring the faces
results = face_recognition.compare_faces([encodeElon],encodeTest)
results_2 = face_recognition.compare_faces([encodeElon],encodeBillTest)


##the best match , the lower the distance the better the match is
faceDist = face_recognition.face_distance([encodeElon],encodeTest)
faceDist_2 = face_recognition.face_distance([encodeElon],encodeBillTest)

##write the results on the test pictures
cv2.putText(imgTest , f'{results[0]} {round(faceDist[0],3)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
cv2.putText(imgBillTest , f'{results_2[0]} {round(faceDist_2[0],3)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )

##output the images
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.imshow('Bill Test',imgBillTest)
cv2.waitKey(0)
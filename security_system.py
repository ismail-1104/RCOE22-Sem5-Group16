import cv2
import winsound

#Import the required libraries
from tkinter import *
from tkinter import ttk

from importlib.resources import path
import numpy as np
import face_recognition
import os

cam = cv2.VideoCapture(0)

#Function for motion detection
def faceRecognition():
    path = 'images'
    image = []
    classNames = []

    myList = os.listdir(path)

    print(myList)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        image.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print(classNames)

    def findEncodings(image):
        encodeList = []
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(image)
    print('Encoding Completed...')

    while True:
        success, img = cam.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

            print(faceDis)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            else:
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
                cv2.putText(img,'Unknown Person',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

                

        cv2.imshow('Webcam',img)
        cv2.waitKey(1)

#Face recognition

#Function for face recognition
# def face-recognition():
    

win = Tk()

win.geometry("700x350")

def open_face_recognition():
   faceRecognition()

#Create a Label widget
Label(win, text= "Click to open a Face Recognition camera").pack(pady=15)

#Create a Button for opening a dialog Box
ttk.Button(win, text= "Face Recognition", command= open_face_recognition).pack()

win.mainloop()
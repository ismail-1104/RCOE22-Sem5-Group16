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
#motion detection function
def motion():
    while cam.isOpened():
        ret,frame1 = cam.read()
        ret,frame2 = cam.read()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
        if cv2.waitKey(10) == ord('a'):
            break
        cv2.imshow('Security Cam', frame1)




#Function for Face Recogonition
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
                winsound.PlaySound('alert.wav'), winsound.SND_ASYNC
                

        cv2.imshow('Webcam',img)
        cv2.waitKey(1)
#tkinter window    
win = Tk()
win.title("Security Camera")
win.geometry("700x350")


def motion_button():
    motion()

def open_face_recognition():
    faceRecognition()


#face recognition
Label(win, text= "Click to open a Face Recognition camera").pack(pady=15)
ttk.Button(win, text= "Face Recognition", command= open_face_recognition).pack()

#motion detection
Label(win, text= "Click to open a Motion Detection Camera").pack(pady=15)
ttk.Button(win, text= "Motion Detection", command= motion_button).pack()



win.mainloop()
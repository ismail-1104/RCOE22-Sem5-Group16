from importlib.resources import path
import cv2
import numpy as np
import face_recognition
import os
import winsound

# imgIsmail = face_recognition.load_image_file('images/IMG_20220710_091836.jpg')
# imgIsmail = cv2.cvtColor(imgIsmail,cv2.COLOR_BGR2RGB)

cam = cv2.VideoCapture(0)

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
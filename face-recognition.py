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


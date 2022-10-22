import cv2
from numpy import diff
cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    if cv2.waitKey(10) == ord('e'):
        break
    cv2.imshow('Security Camera', diff)
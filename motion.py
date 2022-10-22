from turtle import color
import cv2
from numpy import diff
cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    color, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    if cv2.waitKey(10) == ord('e'):
        break
    cv2.imshow('Security Camera', blur)
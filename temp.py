import cv2
import numpy as np
cam = cv2.VideoCapture(0)
while(True):
	ret,frame = cam.read()
	#frame = cv2.blur(frame,(5,5))
	hsvImg = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	hsvImg = cv2.blur(hsvImg,(50,50))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cv2.imshow('HSV Image',hsvImg)
import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

import sys # for input for user file   from user
import mnist_loader
import Network as network


cap = cv2.VideoCapture(0)
#kernal=np.ones( (5,5), np.float32)/25
ret3 =200
#initialize network
net = network.Network([784, 30,20, 10])  #input,hidden,output
#net.SGD(training_data, 5, 10, 2.0, test_data = test_data)
f=open('bw.784.30.20.10.bin','rb')
net.biases = np.load(f)	
net.weights = np.load(f)	 # load trained weights  and biases
f.close() #close the file after reading weights and biases
	
while (True):
	ret, frame = cap.read()
	grayOrig = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#gray = cv2.filter2D(grayOrig,-1,kernal)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
	gray = cv2.GaussianBlur(grayOrig,(5,5),0)
	gray = cv2.equalizeHist(grayOrig)
	#normal thresholding
	ret,thresh1 =cv2.threshold(gray,25,255,cv2.THRESH_BINARY)
	
	#thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#ret, thresh1 = cv2.threshold(gray,125,255,cv2.THRESH_OTSU)
	
	#ret3,thresh1 = cv2.threshold(gray,ret3,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	'''
	# now find contours for image ' thresh1'
	contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	areaArray = []
	count = 1
	for i, c in enumerate(contours):
		area = cv2.contourArea(c)
		areaArray.append(area)
	#first sort the array by area
	sorteddata = sorted(zip(areaArray, contours), key = lambda x: x[0], reverse = True)
	#find the nth largest contour [n-1][1], in this case 2
	
	if len(sorteddata)>1:
		largestcontour = sorteddata[0][1]
	
	#draw it
	x, y, w, h = cv2.boundingRect(largestcontour)
	imgCropped =gray[y:y+h,x:x+w]
	#cv2.imshow('imgCropped', imgCropped)
	cv2.drawContours(gray,largestcontour,-1, (255,0,0),2)
	
	'''
	
	#cv2.imshow('Grayscaled Input Image', gray)
	h, w, = thresh1.shape
	
	imageBlock = thresh1[h/4:3*h/4, w/4:w/4+h/2]
	resizedImage = cv2.resize(imageBlock,(28,28),interpolation =cv2.INTER_LINEAR )
	#make rectangle on the area
	
	cv2.rectangle(thresh1, (w/4,h/4),(w/4+h/2, 3*h/4), (0,255,0),2)
	cv2.imshow('thresholded Image',thresh1)
	
	#resizedImage = cv2.threshold(resizedImage,5,255,cv2.THRESH_BINARY)
	#cv2.imshow('resized image', resizedImage)
	rImg2 = cv2.resize(resizedImage,(320,320), interpolation = cv2.INTER_LINEAR)
	
	cv2.imshow('28x28 img shown', rImg2)
	
	#image testing -- takes the input image and classifies it
	img = resizedImage
	img_array = np.asarray(img)
	img_array = img_array.ravel()
	img_array =1-(1.0/255)*img_array
	img_list = img_array.tolist()
	img_list = np.array(zip(*[img_list]))
	result = net.feedforward(img_list)
	recognizedNum=np.argmax(result)
	confidence = result[recognizedNum][0]*100
	if confidence >99 :
		print 'recognized as '+str(recognizedNum)+' with confidence % of'+str(confidence)
	else : 
		#print 'got as'+str(recognizedNum)+'with confidence'+str(confidence)
		x =1
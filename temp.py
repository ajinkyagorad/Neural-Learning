import numpy as np
import cv2
import Network_v as network
import time
if __name__ == "__main__":
	net = network.Network([784,50,10])
	f = open('50hl.bin','rb')
	net.biases = np.load(f)	
	net.weights = np.load(f)	 # load trained weights  and biases
	f.close() #close the file after reading weights and biases
	weights = net.weights[0]
	print weights
	min,max,minPos,maxPos = cv2.minMaxLoc(weights)
	weiImg = (weights-min)*255/(max-min)
	weiImg = weiImg.astype(int)
#	weiImg = np.zeros(weiImg.shape)
	print weiImg
	cv2.imshow('weights',weiImg)
	time.sleep(5)
	#if cv2.waitKey(1) & 0xFF ==ord('q'):
		
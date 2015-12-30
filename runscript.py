import mnist_loader
import Network as network
import numpy as np
if __name__ == '__main__':
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


	net = network.Network([784, 50, 10])
	net.SGD(training_data, 40 , 10, 3.0, test_data = test_data)
	# to save weights and biases to file
	f = open('50hl.bin','wb')
	np.save(f,net.biases) 
	np.save(f,net.weights)
	f.close()
	#'''

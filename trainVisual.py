import mnist_loader
import Network_v as network
import numpy as np
import sys # for input argument from user
if __name__ == '__main__':
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	layers = [784]	
	for x in sys.argv[1:]:
		layers.append(int(x))
	layers.append(10)
	print 'Given Layers as ' 
	print layers
	net = network.Network(layers)
	net.SGD(training_data, 15 , 1, 3.0, test_data = test_data)
	
	# to save weights and biases to file
	fname = "bw"
	for l in layers:
		fname = fname+'.'+str(l)
	f = open(fname+'.bin','wb')
	np.save(f,net.biases) 
	np.save(f,net.weights)
	f.close()
	#'''

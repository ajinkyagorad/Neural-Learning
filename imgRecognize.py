from PIL import Image
import numpy as np
import sys # for input for user file   from user
import mnist_loader
import Network as network



## Functions#
def recognize(img_name):
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	img_list= loadImage(img_name)
	net = network.Network([784, 50, 10])  #input,hidden,output
	#net.SGD(training_data, 5, 10, 2.0, test_data = test_data)
	f=open('50hl.bin','rb')
	net.biases = np.load(f)	
	net.weights = np.load(f)	 # load trained weights  and biases
	f.close() #close the file after reading weights and biases
	#check for image
	return net.feedforward(img_list)
	# wait for some time
	#import time
	#time.sleep(2)

def loadImage(img_name):
	img = Image.open(img_name)
	img = img.convert('L')
	img_array = np.asarray(img)
	img_array = img_array.ravel()
	img_array =1-(1.0/255)*img_array
	img_list = img_array.tolist()
	img_list = np.array(zip(*[img_list]))
	return img_list
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
	
def detectImage(imagenumber):
	imagePath = 'testimage/'+imagenumber+'.png'
	print 'image used is: '+imagePath
	result = recognize(imagePath)
	#num = np.r_[0:10]
	recognizedNum=np.argmax(result)
	#sum = 0.0
	#for x in result:
	#	sum= x**2+sum
	#confidence = (result[recognizedNum][0]**2)*100.0/sum
	print result
	print 'recognized as '+str(recognizedNum)+' with confidence % of'+str(result[recognizedNum][0]*100)
# main
if __name__ == '__main__':
	if len(sys.argv)>1 :
		for imagenumber in sys.argv[1:]:
			detectImage(imagenumber)
	else :
		imagenumber = str(np.random.random_integers(0,9))
		detectImage(imagenumber)

#saving variables to bin file.
'''
f = file("tmp.bin","wb")
np.save(f,a)
np.save(f,b)
np.save(f,c)
f.close()

f = file("tmp.bin","rb")
aa = np.load(f)
bb = np.load(f)
cc = np.load(f)
f.close()
'''

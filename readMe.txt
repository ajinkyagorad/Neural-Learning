More than Half of the files are from https://github.com/mnielsen/neural-networks-and-deep-learning.git

referenced http://neuralnetworksanddeeplearning.com/chap1.html

Extra (important) files by me are :
50hl.bin  : contains  weights and biases  for MNIST images trained data ( with 50 neurons in hidden layer)

tmp.bin : temporary variable storage

runscript.py : trains the neural network according to mnist data and stores weights and biases to corresponding .bin file

imgRecognize.py : can identify the images (28x28 greyscale) in testimage folder; input file is given in terminal by just the name of the png file ( for eg: 'myFile.png' >> imgRecognize.py myFile)

imgCaptureRecognize.py : identifies the image by input from camera

To RUN : 
* runscript.py  : >>runscript.py 
* recCaptureRecognize.py : make sure u have camera attached to PC., run in console as >>recCaptureRecognize.py
* imgRecognize.py : >>imgRecognize.py <input file names without extension in testimage folder seperated by whitespace>
        ( if file input not given, takes a random file from 0-9 and shows it output )

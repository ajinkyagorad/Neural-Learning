import mnist_loader


if __name__=="__main__":
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	
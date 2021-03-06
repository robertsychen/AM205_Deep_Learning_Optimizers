from tensorflow.examples.tutorials.mnist import input_data

#Pre-process and load MNIST image data.

#training: 55,000 examples. validation: 5,000 examples. testing: 10,000 examples.

def get_mnist_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_dataset = mnist.train.images.reshape((55000,28*28))
    train_labels = mnist.train.labels
    valid_dataset = mnist.validation.images.reshape((5000,28*28))
    valid_labels = mnist.validation.labels
    test_dataset = mnist.test.images.reshape((10000,28*28))
    test_labels = mnist.test.labels
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
import time
import numpy as np

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()
#training: 55,000 examples. validation: 5,000 examples. testing: 10,000 examples.

def run_single_set(num_runs, num_hidden_layers, num_hidden_nodes, optimizer_type, optimizer_params=None):
    train_times = []
    test_accuracies = []

    network = NeuralNetwork(image_size = 28, 
                              num_labels = 10,
                              batch_size = 100,
                              num_hidden_layers = num_hidden_layers,
                              num_hidden_nodes = num_hidden_nodes,
                              train_dataset = train_dataset, 
                              train_labels = train_labels, 
                              valid_dataset = valid_dataset, 
                              valid_labels = valid_labels,
                              optimizer_type = optimizer_type,
                              optimizer_params= optimizer_params)

    for i in xrange(num_runs):
        t = time.time()    
        network.train(num_steps=step, variable_storage_file_name='model0', verbose=True)
      elapsed_time = time.time() - t
      times.append(elapsed_time)
      accuracy, _ = network.test(variable_storage_file_name = 'model0', new_dataset=test_dataset, new_labels=test_labels)
      accuracies.append(accuracy)



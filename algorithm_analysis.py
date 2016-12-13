from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
import time
import numpy as np

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()
#training: 55,000 examples. validation: 5,000 examples. testing: 10,000 examples.

def run_single_set(num_runs, num_hidden_layers, num_hidden_nodes, auto_terminate_num_iter, optimizer_type, optimizer_params=None, noise_type_train=None, noise_mean_train=None, noise_type_test=None, noise_mean_test=None):
  train_times = []
  test_accuracies = []
  num_steps = []

  network = NeuralNetwork(image_size = 28, 
                            num_labels = 10,
                            batch_size = 100,
                            num_hidden_layers = num_hidden_layers,
                            num_hidden_nodes = num_hidden_nodes,
                            train_dataset = train_dataset, 
                            train_labels = train_labels, 
                            valid_dataset = valid_dataset, 
                            valid_labels = valid_labels,
                            test_dataset = test_dataset, 
                            test_labels = test_labels,
                            optimizer_type = optimizer_type,
                            optimizer_params= optimizer_params)

  for i in xrange(num_runs):
    print "Iteration " + str(i)
    t = time.time()    
    _, steps = network.train(auto_terminate_num_iter=auto_terminate_num_iter, variable_storage_file_name='model0', verbose=True, noise_type=noise_type_train, noise_mean=noise_mean_test)
    #_, steps = network.train(num_steps=auto_terminate_num_iter, variable_storage_file_name='model0', verbose=True, noise_type=noise_type_train, noise_mean=noise_mean_test)
    elapsed_time = time.time() - t
    accuracy, _ = network.test(variable_storage_file_name = 'model0', new_dataset=test_dataset, new_labels=test_labels, noise_type=noise_type_test, noise_mean=noise_mean_test)
    train_times.append(elapsed_time)
    test_accuracies.append(accuracy)
    num_steps.append(steps)

  train_times_enhanced = [train_times, np.asarray(train_times).mean(), np.asarray(train_times).std()]
  test_accuracies_enhanced = [test_accuracies, np.asarray(test_accuracies).mean(), np.asarray(test_accuracies).std()]
  num_steps_enhanced = [num_steps, np.asarray(num_steps).mean(), np.asarray(num_steps).std()]

  return train_times_enhanced, test_accuracies_enhanced, num_steps_enhanced

print run_single_set(10, 1, 1024, 100, 'CustomGradientDescent', optimizer_params={'learning_rate':0.05})





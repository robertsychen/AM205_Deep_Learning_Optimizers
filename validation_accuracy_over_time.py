from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
import time
import numpy as np
import cPickle as pickle

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()
#training: 55,000 examples. validation: 5,000 examples. testing: 10,000 examples.

optimizer_parameters = {}
optimizer_parameters['OriginalGradientDescent'] = {'learning_rate': 0.5}
optimizer_parameters['CustomGradientDescent'] = {'learning_rate': 0.5}
optimizer_parameters['OriginalAdam'] = {}
optimizer_parameters['CustomAdam'] = {}
optimizer_parameters['LBFGS'] = {'max_hist': 1000}
optimizer_parameters['ConjugateGradient'] = {'learning_rate': 0.0001, 'min_step': 0.02}
optimizer_parameters['HessianFree'] = {}

def get_single_history(num_runs, num_hidden_layers, num_hidden_nodes, num_steps, optimizer_type, optimizer_params=None, noise_type_train=None, noise_mean_train=None, noise_type_test=None, noise_mean_test=None):

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

       network.train(num_steps=num_steps, variable_storage_file_name='model0', verbose=True, noise_type=noise_type_train, noise_mean=noise_mean_test)
       return network.validation_accuracy_history

diff_algorithms = ['ConjugateGradient', 'HessianFree', 'LBFGS', 'CustomGradientDescent', 'CustomAdam']
diff_noises = [[None, None, None, None],[None, None, 'normal', 0.1],['normal', 0.1, 'normal', 0.1]]

####

steps = 500
num_runs_per = 1

for noise in diff_noises:
       for algo in diff_algorithms:
              for i in xrange(num_runs_per):
                     print noise
                     print algo
                     print 'Starting ' + str(i)
                     result = get_single_history(10, 1, 256, steps, algo, optimizer_parameters[algo], noise[0], noise[1], noise[2], noise[3])
                     print 'Save result '
                     pickle.dump(result, open(algo + str(steps) + noise + str(i) + ".pkl", "wb"))


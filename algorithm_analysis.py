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

def run_single_set(num_runs, num_hidden_layers, num_hidden_nodes, auto_terminate_num_iter, optimizer_type, optimizer_params=None, noise_type_train=None, noise_mean_train=None, noise_type_test=None, noise_mean_test=None):
  train_times = []
  test_accuracies = []
  num_steps = []

  for i in xrange(num_runs):
    print "Iteration " + str(i)

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

    t = time.time()    
    _, steps = network.train(auto_terminate_num_iter=auto_terminate_num_iter, variable_storage_file_name='model0', verbose=True, noise_type=noise_type_train, noise_mean=noise_mean_test)
    #_, steps = network.train(num_steps=auto_terminate_num_iter, variable_storage_file_name='model0', verbose=True, noise_type=noise_type_train, noise_mean=noise_mean_test)
    elapsed_time = time.time() - t
    accuracy, _ = network.test(variable_storage_file_name = 'model0', new_dataset=0, new_labels=0, noise_type=noise_type_test, noise_mean=noise_mean_test)
    train_times.append(elapsed_time)
    test_accuracies.append(accuracy)
    num_steps.append(steps)

  train_times_enhanced = [train_times, np.asarray(train_times).mean(), np.asarray(train_times).std()]
  test_accuracies_enhanced = [test_accuracies, np.asarray(test_accuracies).mean(), np.asarray(test_accuracies).std()]
  num_steps_enhanced = [num_steps, np.asarray(num_steps).mean(), np.asarray(num_steps).std()]

  return train_times_enhanced, test_accuracies_enhanced, num_steps_enhanced

#################################################################


number_name = 0
number_start = 0
number_end = 44

diff_structure = [[1,256], [1,16]]
#diff_algorithms = ['ConjugateGradient', 'HessianFree', 'LBFGS', 'CustomGradientDescent', 'CustomAdam']
diff_algorithms = ['CustomGradientDescent', 'CustomAdam']
diff_noises = [[None, None, None, None],[None, None, 'normal', 0.1],['normal', 0.1, 'normal', 0.1]]

for noise in diff_noises:
  for structure in diff_structure:
    for algo in diff_algorithms:
      if number_name >= number_start and number_name <= number_end:
        if structure == [1,256]:
          result = run_single_set(10, structure[0], structure[1], 50, algo, optimizer_parameters[algo], noise[0], noise[1], noise[2], noise[3])
          print result
          pickle.dump(result, open("newnewresult" + str(number_name) + ".pkl", "wb"))
      number_name += 1





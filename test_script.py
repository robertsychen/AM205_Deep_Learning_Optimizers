from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
from adam_optimizer_test_class import AdamOptimizerTest
import time
import numpy as np

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()

optimizers = [
              ('OriginalGradientDescent', {'learning_rate': 0.5}),
              ('Adam', {}), 
              ('LBFGS', {'max_hist': 1000}),
              ('ConjugateGradient', {'learning_rate': 0.0001, 'min_step': 0.02})
             ]

steps = [10, 20, 50, 100]
stats = {}

for opt in optimizers:
  new_stat = {}
  new_stat['avg_time'] = []
  new_stat['avg_accuracy'] = []
  for step in steps:
    print opt[0]
    print step
    times = []
    accuracies = []
    for i in range(10):
      network = NeuralNetwork(image_size = 28, 
                              num_labels = 10,
                              batch_size = 100,
                              num_hidden_layers = 1,
                              num_hidden_nodes = 100,
                              train_dataset = train_dataset, 
                              train_labels = train_labels, 
                              valid_dataset = valid_dataset, 
                              valid_labels = valid_labels,
                              optimizer_type = opt[0],
                              optimizer_params=opt[1])
      t = time.time()    
      network.train(num_steps=step, variable_storage_file_name='model0', verbose=True)
      elapsed_time = time.time() - t
      times.append(elapsed_time)
      accuracy, _ = network.test(variable_storage_file_name = 'model0')
      accuracies.append(accuracy)
    new_stat['avg_time'].append(np.mean(times))
    new_stat['avg_accuracy'].append(np.mean(accuracies))
  stats[opt[0]] = new_stat
print stats
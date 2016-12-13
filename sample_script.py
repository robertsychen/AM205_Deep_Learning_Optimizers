from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
import time

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()

network = NeuralNetwork(image_size = 28, 
                        num_labels = 10,
                        batch_size = 100,
                        num_hidden_layers = 1,
                        num_hidden_nodes = 100,
                        train_dataset = train_dataset, 
                        train_labels = train_labels, 
                        valid_dataset = valid_dataset, 
                        valid_labels = valid_labels,
                        optimizer_type = 'CustomAdam',
                        #optimizer_type = 'CustomGradientDescent',
                        #optimizer_params={})
                        optimizer_params={'learning_rate':0.5})

#network.train(num_steps = 100, variable_storage_file_name = 'model0', verbose=True, noise_type='normal', noise_mean=0.1)
#network.train(num_steps = 20, variable_storage_file_name = 'model0', verbose=True)
network.train(auto_terminate_num_iter = 5, variable_storage_file_name = 'model0', verbose=True)

#accuracy, _ = network.test(variable_storage_file_name = 'model0', noise_type='normal', noise_mean=0.1)
accuracy, _ = network.test(variable_storage_file_name = 'model0')
print accuracy
from neural_network import NeuralNetwork
from mnist_data import get_mnist_data
from adam_optimizer_test_class import AdamOptimizerTest

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()

network = NeuralNetwork(image_size = 28, 
                        num_labels = 10,
                        batch_size = 100,
                        num_hidden_layers = 1,
                        num_hidden_nodes = 1024,
                        train_dataset = train_dataset, 
                        train_labels = train_labels, 
                        valid_dataset = valid_dataset, 
                        valid_labels = valid_labels,
                        optimizer_type = 'GradientDescent', 
                        optimizer_params ={'learning_rate': 0.5})

network.train(num_steps = 100, variable_storage_file_name = 'model0', verbose=True)

accuracy, _ = network.test(variable_storage_file_name = 'model0')
print accuracy
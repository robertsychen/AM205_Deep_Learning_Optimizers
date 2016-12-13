import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import copy
from optimizers.gradient_descent_op import GradientDescentOpt
from optimizers.bfgs_op import BfgsOpt
from optimizers.conjugate_gradient_op import ConjugateGradientOpt
from optimizers.l_bfgs_op import LBfgsOpt
from optimizers.adam_op import AdamOpt
from optimizers.hessian_free_op import HessianFreeOpt

#note: specifically for image classification, can generalize if we deem necessary
#makes various assumptions about architecture, can alter class as necessary later

class NeuralNetwork(object):
    def __init__(self,
                 image_size, 
                 num_labels, 
                 batch_size, 
                 num_hidden_layers,
                 num_hidden_nodes, 
                 train_dataset, 
                 train_labels, 
                 valid_dataset, 
                 valid_labels,
                 optimizer_type, 
                 optimizer_params=None):
        self.image_size = image_size #number of pixels along side of each image; assumes square images
        self.num_labels = num_labels #total number of classes that images can be assigned to
        self.batch_size = batch_size #number of training images in each mini-batch for optimization
        self.num_hidden_layers = num_hidden_layers #number of hidden layers must be 0 or 1 or 2 (for now)
        self.num_hidden_nodes = num_hidden_nodes #number of hidden nodes per layer; assumes all layers same for now
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.optimizer_type = optimizer_type #string name of optimizer being used
        self.optimizer_params = optimizer_params #dictionary of any parameters needed to run the given optimizer
        self.is_custom_optimizer = None #whether optimizer is built in or is a custom one based on ExternalOptimizerInterface; filled in below
        
        #Set up graph structure.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            
            if num_hidden_layers == 0:
                self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
                self.biases1 = tf.Variable(tf.zeros([num_labels]))
                self.logits = tf.matmul(self.tf_train_dataset, self.weights1) + self.biases1
                self.valid_prediction = tf.nn.softmax(tf.matmul(self.tf_valid_dataset, self.weights1) + self.biases1)
            
            elif num_hidden_layers == 1:
                self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
                self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
                self.biases2 = tf.Variable(tf.zeros([num_labels]))
                self.lay1_train = tf.nn.relu(tf.matmul(self.tf_train_dataset, self.weights1) + self.biases1)
                self.logits = tf.matmul(self.lay1_train, self.weights2) + self.biases2
                self.lay1_valid = tf.nn.relu(tf.matmul(self.tf_valid_dataset, self.weights1) + self.biases1)
                self.valid_prediction = tf.nn.softmax(tf.matmul(self.lay1_valid, self.weights2) + self.biases2)
                
            elif num_hidden_layers == 2:
                self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
                self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes]))
                self.biases2 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
                self.biases3 = tf.Variable(tf.zeros([num_labels]))
                self.lay1_train = tf.nn.relu(tf.matmul(self.tf_train_dataset, self.weights1) + self.biases1)
                self.lay2_train = tf.nn.relu(tf.matmul(self.lay1_train, self.weights2) + self.biases2)
                self.logits = tf.matmul(self.lay2_train, self.weights3) + self.biases3
                self.lay1_valid = tf.nn.relu(tf.matmul(self.tf_valid_dataset, self.weights1) + self.biases1)
                self.lay2_valid = tf.nn.relu(tf.matmul(self.lay1_valid, self.weights2) + self.biases2)
                self.valid_prediction = tf.nn.softmax(tf.matmul(self.lay2_valid, self.weights3) + self.biases3)

            elif num_hidden_layers == 5:
                self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
                self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes]))
                self.biases2 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes]))
                self.biases3 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights4 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes]))
                self.biases4 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights5 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes]))
                self.biases5 = tf.Variable(tf.zeros([num_hidden_nodes]))
                self.weights6 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
                self.biases6 = tf.Variable(tf.zeros([num_labels]))
                self.lay1_train = tf.nn.relu(tf.matmul(self.tf_train_dataset, self.weights1) + self.biases1)
                self.lay2_train = tf.nn.relu(tf.matmul(self.lay1_train, self.weights2) + self.biases2)
                self.lay3_train = tf.nn.relu(tf.matmul(self.lay2_train, self.weights3) + self.biases3)
                self.lay4_train = tf.nn.relu(tf.matmul(self.lay3_train, self.weights4) + self.biases4)
                self.lay5_train = tf.nn.relu(tf.matmul(self.lay4_train, self.weights5) + self.biases5)
                self.logits = tf.matmul(self.lay5_train, self.weights6) + self.biases6
                self.lay1_valid = tf.nn.relu(tf.matmul(self.tf_valid_dataset, self.weights1) + self.biases1)
                self.lay2_valid = tf.nn.relu(tf.matmul(self.lay1_valid, self.weights2) + self.biases2)
                self.lay3_valid = tf.nn.relu(tf.matmul(self.lay2_valid, self.weights3) + self.biases3)
                self.lay4_valid = tf.nn.relu(tf.matmul(self.lay3_valid, self.weights4) + self.biases4)
                self.lay5_valid = tf.nn.relu(tf.matmul(self.lay4_valid, self.weights5) + self.biases5)
                self.valid_prediction = tf.nn.softmax(tf.matmul(self.lay5_valid, self.weights6) + self.biases6)
                
            else:
                raise ValueError('This number of hidden layers not currently supported.')
        
            self.train_prediction = tf.nn.softmax(self.logits)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))
            
            #Select which optimizer to use.
            if optimizer_type == 'OriginalGradientDescent':
                self.is_custom_optimizer = False
                self.optimizer = tf.train.GradientDescentOptimizer(optimizer_params['learning_rate']).minimize(self.loss)
            elif optimizer_type == 'CustomGradientDescent':
                self.is_custom_optimizer = True
                self.optimizer = GradientDescentOpt(loss=self.loss, learning_rate=optimizer_params['learning_rate'])
            elif optimizer_type == 'OriginalAdam':
                self.is_custom_optimizer = False
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            elif optimizer_type == 'CustomAdam':
                self.is_custom_optimizer = True
                self.optimizer = AdamOpt(loss=self.loss, **optimizer_params)
            elif optimizer_type == 'AdamTest': #does the same thing as Adam, but is implemented outside of TensorFlow library
                self.is_custom_optimizer = False
                self.optimizer = AdamOptimizerTest().minimize(self.loss)
            elif optimizer_type == 'BFGS':
                self.is_custom_optimizer = True
                self.optimizer = BfgsOpt(loss=self.loss)
            elif optimizer_type == 'ConjugateGradient':
                self.is_custom_optimizer = True
                self.optimizer = ConjugateGradientOpt(loss=self.loss, line_search_params=optimizer_params)
            elif optimizer_type == 'LBFGS':
                self.is_custom_optimizer = True
                self.optimizer = LBfgsOpt(loss=self.loss, max_hist=optimizer_params['max_hist'])
            elif optimizer_type == 'HessianFree':
                self.is_custom_optimizer = True
                self.optimizer = HessianFreeOpt(loss=self.loss)
            else:
                raise ValueError('Not a valid optimizer type.')
                
    def __accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def __apply_normal_noise(self, data, mean):
        sigma = np.sqrt(np.pi/2.0)*mean
        if sigma == 0.0:
            return data
        else:
            random_values = np.random.normal(0.0, sigma, data.shape[0]*data.shape[1]).reshape(data.shape)
            return np.clip((data + random_values), 0.0, 1.0)

    def train(self, variable_storage_file_name, num_steps=None, auto_terminate_num_iter=None, verbose=True, noise_type=None, noise_mean=None):
        #Exactly one of num_steps or auto_terminate_validation_cutoff should be None, the other should not.
        #if num_steps is not None: stop training after that number of steps
        #if auto_terminate_num_iter is not None: terminate whenever the validation set accuracy has not set a new record in auto_terminate_num_iter iterations

        assert((auto_terminate_num_iter is None) ^ (num_steps is None))
        is_fixed_num_steps = True
        if num_steps is None:
            is_fixed_num_steps = False

        global previous_update_info
        previous_update_info = [0, None, None, None, 0, None] #step number, minibatch loss, minibatch accuracy, validation set accuracy, num. iterations since last validation accuracy improvement, previous best validation accuracy

        verbose_print_freq = None
        if verbose:
            if is_fixed_num_steps:
                verbose_print_freq = min(500,(max(1,num_steps/10)))
            else:
                verbose_print_freq = 1

        def __performance_update_assigner_and_printer(l, predictions, step):
            global previous_update_info
            if step != previous_update_info[0]:

                previous_update_info[0] = step
                previous_update_info[1] = l
                previous_update_info[2] = self.__accuracy(predictions, batch_labels)
                previous_update_info[3] = self.__accuracy(self.valid_prediction.eval(), self.valid_labels)

                if verbose and (step % verbose_print_freq == 0):
                    print("Minibatch loss at step %d: %f" % (previous_update_info[0], previous_update_info[1]))
                    print("Minibatch accuracy: %.1f%%" % previous_update_info[2])
                    print("Validation accuracy: %.1f%%" % previous_update_info[3])       

                #keep track of how many iterations since validation accuracy improved
                if not is_fixed_num_steps:
                    if previous_update_info[5] is None:
                        previous_update_info[5] = previous_update_info[3]
                    elif previous_update_info[3] > previous_update_info[5]:
                        previous_update_info[5] = previous_update_info[3]
                        previous_update_info[4] = 0
                    else:
                        previous_update_info[4] += 1


        with tf.Session(graph=self.graph) as session:

            tf.initialize_all_variables().run(session=session)

            #Add noise (optional).
            this_train_dataset = copy.deepcopy(self.train_dataset)
            if noise_type:
                if noise_type == 'normal':
                    this_train_dataset = self.__apply_normal_noise(this_train_dataset, noise_mean)
                else:
                    raise ValueError('noise type not currently supported')

            step = 0
            while True:
                step += 1

                def __performance_update_wrapper(l, predictions):
                        __performance_update_assigner_and_printer(l, predictions, step)

                index_subset = np.random.choice(self.train_labels.shape[0], size=self.batch_size)
                batch_data = this_train_dataset[index_subset, :]
                batch_labels = self.train_labels[index_subset, :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}

                #different behavior depending on whether optimizer is built-in or is based on ExternalOptimizerInterface
                if self.is_custom_optimizer:
                    self.optimizer.minimize(session, feed_dict=feed_dict, fetches=[self.loss, self.train_prediction], loss_callback=__performance_update_wrapper)
                else:
                    _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                    __performance_update_assigner_and_printer(l, predictions, step)

                if is_fixed_num_steps:
                    if step >= num_steps:
                        break
                else:
                    if previous_update_info[4] >= auto_terminate_num_iter:
                        break
                
            validation_accuracy = self.__accuracy(self.valid_prediction.eval(), self.valid_labels)
            if verbose:
                print("Final Validation accuracy: %.1f%%" % validation_accuracy)
            
            saver = tf.train.Saver() #this saving behavior allows you to train multiple versions of the model using the same class
            save_path = saver.save(session, variable_storage_file_name)



            return validation_accuracy
        
    def test(self, variable_storage_file_name, new_dataset=0, new_labels=0, noise_type=None, noise_mean=None):
        #if new_dataset is 0 and new_labels is 0: uses the validation set and labels to predict and score
        #if new_dataset isnt 0 and new_labels isnt 0: uses the new test set to predict
        #if new_dataset isnt 0 None and new_labels isnt 0: uses the new test set and labels to predict and score
        #important note: new_dataset & new_labels must have same amount of data points as self.valid_dataset, self.valid_labels
        #this is an unfortunate truth based on the tensor setup above

        #Add some noise to test data if applicable
        noisy_dataset = copy.deepcopy(self.valid_dataset)
        if noise_type:
            if new_dataset == 0 and new_labels == 0:
                if noise_type == 'normal':
                    noisy_dataset = self.__apply_normal_noise(noisy_dataset, noise_mean)
                else: 
                    raise ValueError('Noise type not supported')
            else:
                raise ValueError('No such functionality to add noise to testing data')

        resulting_accuracy = None
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            saver.restore(session, variable_storage_file_name)    
            
            this_dataset = None
            this_labels = None
            if type(new_dataset) is not int: #checking if it is 0 or if is actual numpy array user inputted data
                this_dataset = new_dataset
                this_labels = new_labels
            else:
                this_dataset = self.valid_dataset
                this_labels = self.valid_labels

            #Switch to noisy data if applicable
            if noise_type:
                this_dataset = noisy_dataset

            feed_dict_clean = {self.tf_valid_dataset: this_dataset}
            valid_y = session.run(self.valid_prediction, feed_dict=feed_dict_clean)
            resulting_accuracy = None if type(new_dataset) is not int and type(new_labels) is int else self.__accuracy(valid_y, this_labels)
        return resulting_accuracy, valid_y
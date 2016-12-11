import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import copy
from optimizers.gradient_descent_op import GradientDescentOpt
from optimizers.bfgs_op import BfgsOpt
from optimizers.conjugate_gradient_op import ConjugateGradientOpt

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
            elif optimizer_type == 'Adam':
                self.is_custom_optimizer = False
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            elif optimizer_type == 'AdamTest': #does the same thing as Adam, but is implemented outside of TensorFlow library
                self.is_custom_optimizer = False
                self.optimizer = AdamOptimizerTest().minimize(self.loss)
            elif optimizer_type == 'BFGS':
                self.is_custom_optimizer = True
                self.optimizer = BfgsOpt(loss=self.loss)
            elif optimizer_type == 'ConjugateGradient':
                self.is_custom_optimizer = True
                self.optimizer = ConjugateGradientOpt(loss=self.loss, line_search_params=optimizer_params)
            else:
                raise ValueError('Not a valid optimizer type.')
                
    def __accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def train(self, num_steps, variable_storage_file_name, verbose=True):
        #currently computes how the validation set is doing over time as well
        #could add functionality to turn this off        

        def __performance_update_printer(l, predictions):
            if verbose and (step % (max(1,num_steps/10)) == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % self.__accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % self.__accuracy(self.valid_prediction.eval(), self.valid_labels))

        with tf.Session(graph=self.graph) as session:
            # for var in tf.trainable_variables():
            #     print var
            # tf.variables_initializer(tf.trainable_variables()).run(session=session)
            tf.initialize_all_variables().run(session=session)
            for step in range(num_steps):
                index_subset = np.random.choice(self.train_labels.shape[0], size=self.batch_size)
                batch_data = self.train_dataset[index_subset, :]
                batch_labels = self.train_labels[index_subset, :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}

                #different behavior depending on whether optimizer is built-in or is based on ExternalOptimizerInterface
                if self.is_custom_optimizer:
                    print ('hooo')
                    self.optimizer.minimize(session, feed_dict=feed_dict, fetches=[self.loss, self.train_prediction], loss_callback=__performance_update_printer)
                else:
                    _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                    __performance_update_printer(l, predictions)
            
            validation_accuracy = self.__accuracy(self.valid_prediction.eval(), self.valid_labels)
            if verbose:
                print("Final Validation accuracy: %.1f%%" % validation_accuracy)
            
            saver = tf.train.Saver() #this saving behavior allows you to train multiple versions of the model using the same class
            save_path = saver.save(session, variable_storage_file_name)
            return validation_accuracy
        
    def test(self, variable_storage_file_name, new_dataset=0, new_labels=0):
        #if new_dataset is 0 and new_labels is 0: uses the validation set and labels to predict and score
        #if new_dataset isnt 0 and new_labels isnt 0: uses the new test set to predict
        #if new_dataset isnt 0 None and new_labels isnt 0: uses the new test set and labels to predict and score
        #important note: new_dataset & new_labels must have same amount of data points as self.valid_dataset, self.valid_labels
        #this is an unfortunate truth based on the tensor setup above
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
            
            feed_dict_clean = {self.tf_valid_dataset: this_dataset}
            valid_y = session.run(self.valid_prediction, feed_dict=feed_dict_clean)
            resulting_accuracy = None if type(new_dataset) is not int and type(new_labels) is int else self.__accuracy(valid_y, this_labels)
        return resulting_accuracy, valid_y
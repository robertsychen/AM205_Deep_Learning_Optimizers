from optimizers.gradient_descent_op import GradientDescentOpt
from optimizers.bfgs_op import BfgsOpt
from optimizers.conjugate_gradient_op import ConjugateGradientOpt
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from adam_optimizer_test_class import AdamOptimizerTest
import tensorflow as tf

vector2 = tf.Variable([5., 7.], 'vector2')
  # Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector2))
optimizer = ConjugateGradientOpt(loss, min_step=0.0001, 
                                       line_search_params={'learning_rate': 0.0001, 'min_step': 0.02})

with tf.Session() as session:
  tf.initialize_variables([vector2]).run()
  result = optimizer.minimize(session)
  print result
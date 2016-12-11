from optimizers.gradient_descent_op import GradientDescentOpt
from optimizers.bfgs_op import BfgsOpt
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from adam_optimizer_test_class import AdamOptimizerTest
import tensorflow as tf

#This is old and won't work now that the opt functions are changed.

vector2 = tf.Variable([5, 2.], 'vector2')
  # Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector2))
optimizer = GradientDescentOpt(loss, min_step=0.0001, learning_rate=0.2)
#optimizer = BfgsOpt(loss, min_step=0.1)

with tf.Session() as session:
  tf.initialize_variables([vector2]).run()
  _ = optimizer.minimize(session)
  print vector2.eval()
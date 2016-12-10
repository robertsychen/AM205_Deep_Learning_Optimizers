from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class GradientDescentOpt(ExternalOptimizerInterface):
  def __init__(self, loss, min_step, learning_rate):
    """Initialize a new interface instance.
    """
    super(GradientDescentOpt, self).__init__(loss)
    self.min_step = min_step
    self.learning_rate = learning_rate

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    epsilon = float('inf')
    current_val = initial_val
    while epsilon > self.min_step:
      
      print(current_val.shape)
      _, grad = loss_grad_func(current_val)
      delta = - grad * self.learning_rate
      new_val = current_val + delta
      epsilon = np.linalg.norm(new_val - current_val)
      current_val = new_val
      print(current_val)
    return current_val
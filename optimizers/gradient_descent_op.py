from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class GradientDescentOpt(ExternalOptimizerInterface):
  def __init__(self, loss, learning_rate):
    """Initialize a new interface instance.
    """
    super(GradientDescentOpt, self).__init__(loss)
    self.learning_rate = learning_rate

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    current_val = initial_val
    _, grad = loss_grad_func(current_val)
    delta = - grad * self.learning_rate
    new_val = current_val + delta

    return new_val
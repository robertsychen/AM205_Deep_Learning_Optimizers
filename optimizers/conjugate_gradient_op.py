from __future__ import division
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class ConjugateGradientOpt(ExternalOptimizerInterface):
  def __init__(self, loss, min_step, line_search_params):
    """Initialize a new interface instance.
    """
    super(ConjugateGradientOpt, self).__init__(loss)
    self.min_step = min_step
    self.line_search_params = line_search_params

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    epsilon = float('inf')
    
    _, grad = loss_grad_func(initial_val)
    s = - grad
    current_val = initial_val
    while epsilon > self.min_step:

      # Gradient Descent for Line Search
      epsilon2 = float('inf')
      learning_rate = self.line_search_params['learning_rate']
      eta = 0

      def line_func_loss(x):
        loss, _ = loss_grad_func(current_val + x * s)
        return loss

      def line_func_grad(x):
        h = 0.001
        line_grad = (line_func_loss(x + h) - line_func_loss(x - h)) / (2 * h)
        return line_grad

      while epsilon2 > self.line_search_params['min_step']:
        eta_new = eta - learning_rate * line_func_grad(eta)
        epsilon2 = np.linalg.norm(eta_new - eta)
        eta = eta_new

      # Conjugate Gradient Algorithm
      new_val = current_val + eta * s
      _, new_grad = loss_grad_func(new_val)
      beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
      s = - new_grad + beta * s
      grad = new_grad
      epsilon = np.linalg.norm(new_val - current_val)
      current_val = new_val
      print(current_val)
    return current_val
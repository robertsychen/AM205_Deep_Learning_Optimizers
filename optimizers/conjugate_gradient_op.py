from __future__ import division
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class ConjugateGradientOpt(ExternalOptimizerInterface):
  def __init__(self, loss, line_search_params):
    """Initialize a new interface instance.
    """
    super(ConjugateGradientOpt, self).__init__(loss)
    self.line_search_params = line_search_params

    self.grad = None
    self.s = None

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):

    current_val = initial_val

    if self.grad is None:
      _, self.grad = loss_grad_func(current_val)
      self.s = - self.grad

    # Gradient Descent for Line Search
    epsilon2 = float('inf')
    learning_rate = self.line_search_params['learning_rate']
    eta = 0

    def line_func_loss(x):
      this_loss, _ = loss_grad_func(current_val + x * self.s)
      return this_loss

    def line_func_grad(x):
      h = 0.001
      line_grad = (line_func_loss(x + h) - line_func_loss(x - h)) / (2 * h)
      return line_grad

    while epsilon2 > self.line_search_params['min_step']:
      eta_new = eta - learning_rate * line_func_grad(eta)
      epsilon2 = np.linalg.norm(eta_new - eta)
      eta = eta_new

    # Conjugate Gradient Algorithm
    new_val = current_val + eta * self.s
    _, new_grad = loss_grad_func(new_val)
    beta = np.dot(new_grad, new_grad) / np.dot(self.grad, self.grad)
    new_s = - new_grad + beta * self.s
    self.grad = new_grad
    self.s = new_s
    return new_val
from __future__ import division
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class BfgsOpt(ExternalOptimizerInterface):
  def __init__(self, loss, min_step):
    """Initialize a new interface instance.
    """
    super(BfgsOpt, self).__init__(loss)
    self.min_step = min_step

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    epsilon = float('inf')
    current_val = initial_val
    n = len(initial_val)
    eye = np.identity(n)
    hess = eye
    while epsilon > self.min_step:
      _, grad = loss_grad_func(current_val)
      delta = - np.matmul(hess, grad)
      new_val = current_val + delta
      print(current_val)
      _, new_grad = loss_grad_func(new_val)
      y = new_grad - grad
      print(y)
      print(delta)
      print('\n')
      denom = np.dot(y, delta)
      if denom != 0:
        rho = 1 / np.dot(y, delta)
      else:
        rho = 0
      first_term = eye - np.outer(delta, rho * y)
      second_term = eye - rho * np.outer(y, delta) 
      hess =  np.matmul(np.matmul(first_term, hess), second_term) + rho * np.outer(delta, delta)
      epsilon = np.linalg.norm(new_val - current_val)
      current_val = new_val
    return current_val
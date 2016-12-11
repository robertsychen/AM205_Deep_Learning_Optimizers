from __future__ import division
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class BfgsOpt(ExternalOptimizerInterface):
  def __init__(self, loss):
    """Initialize a new interface instance.
    """
    super(BfgsOpt, self).__init__(loss)

    self.hess = None

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    current_val = initial_val
    n = len(initial_val)
    eye = np.identity(n)

    if self.hess is None:
      self.hess = eye

    print (self.hess)
    print (self.hess.shape)

    _, grad = loss_grad_func(current_val)
    delta = - np.matmul(self.hess, grad)
    new_val = current_val + delta
    _, new_grad = loss_grad_func(new_val)
    y = new_grad - grad

    denom = np.dot(y, delta)
    if denom != 0:
      rho = 1 / np.dot(y, delta)
    else:
      rho = 0
    first_term = eye - np.outer(delta, rho * y)
    second_term = eye - rho * np.outer(y, delta) 
    new_hess =  np.matmul(np.matmul(first_term, self.hess), second_term) + rho * np.outer(delta, delta)
    self.hess = new_hess
    return new_val
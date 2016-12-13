from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class HessianFreeOpt(ExternalOptimizerInterface):
  def __init__(self, loss, n_directs=5):
    """Initialize a new interface instance.
      n_direct: number of directions to search; increasing this number will increase training accuracy,
        but also increase running time
    """
    self.n_directs = n_directs
    super(HessianFreeOpt, self).__init__(loss)

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    _, old_grad = loss_grad_func(initial_val)
    direct = - old_grad
    n = len(initial_val)
    directs_explored = 1
    current_val = initial_val
    while directs_explored < self.n_directs:
      # Update position
      hess_curr_val = self._approx_hess_vec_prod(loss_grad_func, current_val, current_val)
      hess_direct = self._approx_hess_vec_prod(loss_grad_func, current_val, direct)
      alpha = - np.dot(direct, hess_curr_val + old_grad) / np.dot(direct, hess_direct)
      current_val = current_val + alpha * direct

      # Update direction
      _, grad = loss_grad_func(current_val)
      beta = np.dot(grad, hess_direct) / np.dot(direct, hess_direct)
      direct = - grad + beta * direct
      directs_explored += 1
    return current_val    

  def _approx_hess_vec_prod(self, loss_grad_func, current_val, vec, epsilon=10**-4):
    _, grad_new = loss_grad_func(current_val + epsilon * vec)
    _, grad_old = loss_grad_func(current_val)
    return (grad_new - grad_old) / epsilon

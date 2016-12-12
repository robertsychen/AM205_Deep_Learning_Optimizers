from __future__ import division
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
from collections import deque
import numpy as np
import tensorflow as tf

class LBfgsOpt(ExternalOptimizerInterface):
  def __init__(self, loss, max_hist):
    """Initialize a new interface instance.
    """
    super(LBfgsOpt, self).__init__(loss)
    self.max_hist = max_hist
    self.deltas = deque([])
    self.delta_grads = deque([])
    self.rhos = deque([])

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    
    # Two-Loop Recursion
    old_loss, grad = loss_grad_func(initial_val)
    q = grad
    alphas = deque([])
    for delta, delta_grad, rho in reversed(zip(self.deltas, self.delta_grads, self.rhos)):
        alpha = rho * np.dot(delta, q)
        q = q - alpha * delta_grad
        alphas.appendleft(alpha)

    if self.deltas and self.delta_grads:
        h0 = np.dot(self.deltas[-1], self.delta_grads[-1]) / \
         np.dot(self.delta_grads[-1], self.delta_grads[-1])
    else:
        h0 = 1

    z = h0 * q

    for delta, delta_grad, rho, alpha in zip(self.deltas, self.delta_grads, self.rhos, alphas):
        beta = rho * np.dot(delta_grad, z)
        z = z + (alpha - beta) * delta

    new_direct = -z

    # Line Search 
    step_len = 1
    max_step = 10
    num_step = 0
    c1 = 10^-4
    c2 = 0.9
    cond1 = False
    cond2 = False

    while (not cond1 or not cond2) and (num_step < max_step):
        num_step += 1
        new_delta = step_len * new_direct
        new_val = initial_val + new_delta
        new_loss, new_grad = loss_grad_func(new_val)
        cond1 = (new_loss <= old_loss + c1 * step_len * np.dot(new_direct, grad))
        cond2 = (np.dot(new_direct, new_grad) >= c2 * np.dot(new_direct, grad))
        step_len = step_len / 5

    new_delta_grad = new_grad - grad
    new_rho = 1 / np.dot(new_delta_grad, new_delta)

    # Update History
    hist_size = len(self.deltas)
    if hist_size >= self.max_hist:
        self.deltas.popleft()
        self.delta_grads.popleft()
        self.rhos.popleft()

    self.deltas.append(new_delta)
    self.delta_grads.append(new_delta_grad)
    self.rhos.append(new_rho)
    
    return new_val
from tensorflow.contrib.opt.python.training.external_optimizer import ExternalOptimizerInterface
import numpy as np
import tensorflow as tf

class AdamOpt(ExternalOptimizerInterface):
  def __init__(self, loss, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10**-8):
    """Initialize a new interface instance.
    """
    super(AdamOpt, self).__init__(loss)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m = None
    self.v = None
    self.t = 0 

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    if self.t == 0:
        n = len(initial_val)
        self.m = np.zeros(n)
        self.v = np.zeros(n)

    self.t += 1
    _, grad = loss_grad_func(initial_val)
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
    m_hat = self.m / (1 - (self.beta1 ** self.t))
    v_hat = self.v / (1 - (self.beta2 ** self.t))
    new_val = initial_val - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    return new_val
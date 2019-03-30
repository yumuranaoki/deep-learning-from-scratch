import sys
import os
import numpy as np

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))

from common.layer import MatMul, SoftmaxWithLoss
from common.util import processing, create_contexts_target, convert_one_hot

class SimpleCBOW:
  def __init__(self, vocab_size, hidden_size):
    W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
    W_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype('f')

    self.input_layor0 = MatMul(W_in)
    self.input_layer1 = MatMul(W_in)
    self.out_layer = MatMul(W_out)
    self.loss_layer = SoftmaxWithLoss()

    layers = [self.input_layor0, self.input_layer1, self.out_layer]
    self.params, self.grads = [], []

    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads
    
    self.word_vecs = W_in

  def forward(self, contexts, target):
    h0 = self.input_layer0.forward(contexts[:, 0, :])
    h1 = self.input_layer1.forward(contexts[:, 1, :])
    h = (h0 + h1) / 2

    score = self.out_layer.forward(h)
    loss = self.loss_layer.forward(score, target)
    return loss
  
  def backword(self, dout=1):
    ds = self.loss_layer.backward(dout)
    da = self.out_layer.backword(ds)
    da /= 2
    self.input_layer0.backward(da)
    self.input_layer1.backword(da)
    return None
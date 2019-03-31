import numpy as np

def sigmoid(x):
  return 1 / 1 + np.exp(-x)

def softmax(x):
  if x.ndim == 2:
    x = x - x.max(axis=1, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=1, keepdims=True)
  elif x.ndim == 1:
    x = x - np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))

  return x

class MatMul:
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.x = None

  def forward(self, x):
    W = self.params
    out = np.dot(x, W)
    self.x = x
    return out

  def backword(self, dout):
    W, = self.params
    dx = np.dot(dout, W.T)
    dW = np.dot(self.x.T, dout)
    self.grads[0][...] = dW
    return dx

class Sigmoid:
  def __init__(self):
    self.params = []

  def forward(self, x):
    return 1 / (1 + np.exp(-x))
  
  def backword(self, dout):
    dx = dout * self.out * (1 - self.out)
    return dx

class Affine:
  def __init__(self, W, b):
    self.params = [W, b]
    self.grads = [np.zeros_like(W), np.zeros_like(b)]
    self.x = None

  def forward(self, x):
    W, b = self.params
    out = np.dot(x, W) + b
    self.x = x
    return out
  
  def backword(self, dout):
    W, b = self.params
    dx = np.dot(dout, W.T)
    dW = np.dot(self.x.T, dout)
    db = np.sum(dout, axis=0)

    self.grads[0][...] = dW
    self.grads[1][...] = db
    return dx   

class Softmax:
  def __init__(self):
    self.params, self.grads = [], []
    self.out = None
  
  def forward(self, x):
    self.out = softmax(x)
    return self.out

  def backword(self, dout):
    dx = self.out * dout
    sumdx = np.sum(dx, axis=1, keepdims=True)
    dx -= self.out * sumdx
    return dx

class SoftmaxWithLoss:
  def __init__(self):
    self.params, self.grads = [], []
    self.y = None  # softmaxの出力
    self.t = None  # 教師ラベル

  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)

    # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
    if self.t.size == self.y.size:
        self.t = self.t.argmax(axis=1)

    loss = cross_entropy_error(self.y, self.t)
    return loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]

    dx = self.y.copy()
    dx[np.arange(batch_size), self.t] -= 1
    dx *= dout
    dx = dx / batch_size

    return dx

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
      
  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size:
    t = t.argmax(axis=1)
            
  batch_size = y.shape[0]

  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Embedding:
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.idx = None
  
  def forward(self, idx):
    W, = self.params
    self.idx = idx
    out = W[idx]
    return out
  
  def backword(self, dout):
    dW, = self.grads
    dW[...] = 0
    for i, word_id in enumerate(self.idx):
      dW[word_id] += dout[i]
    
    return None
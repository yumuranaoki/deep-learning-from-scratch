import numpy as np

class RNN:
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.cache = None
  
  def forward(self, x, h_prev):
    Wh, Wx, b = self.params
    t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
    h_next = np.tanh(t)

    self.cache = (x, h_prev, h_next)
    return h_next

  def backword(self, dh_next):
    Wx, Wh, b = self.params
    x, h_prev, h_next = self.cache

    dt = dh_next * (1 - h_next ** 2)
    db = np.sum(dt, axis=0)
    dWh = np.dot(h_prev.T, dt)
    dh_prev = np.dot(dt, Wh.T)
    dWx = np.dot(x.T, dt)
    dx = np.dot(dt, Wx.T)

    self.grads[0][...] = dWx
    self.grads[1][...] = dWh
    self.grads[2][...] = db

    return dx, dh_prev

class TimeRNN:
  def __init__(self, Wx, Wh, b, stateful=False):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layers = None
    self.h, self.dh = None, None
    self.stateful = stateful

  def set_state(self, h):
    self.h = h

  def reset_state(self, dh):
    self.h = None

  # @params xs: mini batch of input
  def forward(self, xs):
    Wx,Wh, b = self.params
    # N: バッチサイズ
    # T: 時系列の塊
    # D: 分散表現のサイズ
    N, T, D = xs.shape
    D, H = Wx.shape

    self.layers = []
    hs = np.empty((N, T, H), dtype='f')

    # 状態を引き継がない場合は初期化
    if not self.stateful or self.h is None:
      self.h = np.zeros((N, H), dtype='f')

    for t in range(T):
      layer = RNN(*self.params) # unpack
      self.h = layer.forward(xs[:, t, :], self.h)
      hs[:, t, :] = self.h
      self.layers.append(layer)

    return hs

  # @params dhs: mini batch of output gradient
  def backword(self, dhs):
    Wx,Wh, b = self.params
    # N: バッチサイズ
    # T: 時系列の塊
    # H: 隠れ層のサイズ
    # D: 分散表現のサイズ
    N, T, H = dhs.shape
    D, H = Wx.shape

    dxs = np.empty((N, T, D), dtype='f')
    dh = 0
    grads = [0, 0, 0]
    for t in reversed(range(T)):
      layer = self.layers[t]
      dx, dh = layer.backword(dhs[:, t, :])
      dxs[:, t, :] = dx

      for i, grad in enumerate(grads):
        self.grads[i][...] = grad
      self.dh = dh

      return dxs
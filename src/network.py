import numpy as np

class Layer:
    def __init__(self, n_in, n_out, activation = 'tanh'):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b

        if self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'linear':
            self.a = self.z

        return self.a

    def backward(self, delta):
        if self.activation == 'tanh':
            d_act = 1 - np.tanh(self.z) ** 2
        elif self.activation == 'linear':
            d_act = np.ones_like(self.z)

        delta_z = delta * d_act
        self.dW = self.x.T @ delta_z
        self.db = np.sum(delta_z, axis=0)

        return delta_z @ self.W.T

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x , y_true):
        y_pred = self.forward(x)
        return  np.mean((y_pred - y_true) ** 2)

    def backward(self, x , y_true):
        y_pred = self.forward(x)
        delta = (2/len(y_true)) * (y_pred - y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer.W.ravel())
            params.append(layer.b)
        return np.concatenate(params)

    def set_params(self, params):
        idx = 0
        for layer in self.layers:
            w_size = layer.W.size
            b_size = layer.b.size
            layer.W = params[idx:idx + w_size].reshape(layer.W.shape)
            idx += w_size
            layer.b = params[idx:idx + b_size]
            idx += b_size

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads.append(layer.dW.ravel())
            grads.append(layer.db)
        return np.concatenate(grads)


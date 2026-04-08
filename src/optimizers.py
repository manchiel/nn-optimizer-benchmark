import numpy as np
from scipy.optimize import minimize

class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, network, x, y):
        network.backward(x, y)
        grads = network.get_grads()
        params = network.get_params()
        params = params - self.lr * grads
        network.set_params(params)

    def optimize(self, network, x, y, epochs):
        history = []
        for epoch in range(epochs):
            self.step(network, x, y)
            history.append(network.loss(x,y))
        return history



class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, network, x, y):
        network.backward(x, y)
        grads = network.get_grads()
        params = network.get_params()

        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        network.set_params(params)

    def optimize(self, network, x, y, epochs):
        history = []
        for epoch in range(epochs):
            self.step(network, x, y)
            history.append(network.loss(x,y))
        return history


class LevenbergMarquardt:
    def __init__(self, lambda_init=0.01, lambda_factor=3.0):
        self.lam = lambda_init
        self.lambda_factor = lambda_factor

    def _jacobian(self, network, x, eps=1e-5):
        params = network.get_params()
        n_data = x.shape[0]
        n_params = len(params)
        J = np.zeros((n_data, n_params))

        f0 = network.forward(x).ravel()

        for j in range(n_params):
            params_plus = params.copy()
            params_plus[j] += eps
            network.set_params(params_plus)
            f_plus = network.forward(x).ravel()
            J[:, j] = (f_plus - f0) / eps

        network.set_params(params)
        return J

    def step(self, network, x, y):
        params = network.get_params()
        r = network.forward(x).ravel() - y.ravel()
        J = self._jacobian(network, x)

        JtJ = J.T @ J
        Jtr = J.T @ r

        n_params = len(params)
        A = JtJ + self.lam * np.eye(n_params)
        delta = np.linalg.solve(A, -Jtr) #better than inverse matrix

        new_params = params + delta
        network.set_params(new_params)

        new_r = network.forward(x).ravel() - y.ravel()
        old_loss = np.mean(r ** 2)
        new_loss = np.mean(new_r ** 2)

        if new_loss < old_loss:
            self.lam /= self.lambda_factor
        else:
            network.set_params(params)
            self.lam *= self.lambda_factor

    def optimize(self, network, x, y, epochs):
        history = []
        for epoch in range(epochs):
            self.step(network, x, y)
            history.append(network.loss(x, y))
        return history

class LBFGS:
    def __init__(self):
        self.history = []

    def optimize(self, network, x, y, epochs):
        self.history = []
        self._network = network
        self._x = x
        self._y = y

        def loss_fn(params):
            network.set_params(params)
            return network.loss(x, y)

        def grad_fn(params):
            network.set_params(params)
            network.backward(x, y)
            return network.get_grads()

        def callback(params):
            self.history.append(network.loss(x, y))

        result = minimize(fun=loss_fn,
                          x0=network.get_params(),
                          jac=grad_fn,
                          method='L-BFGS-B',
                          callback=callback,
                          options={'maxiter': epochs})

        network.set_params(result.x)
        return self.history


class LMBroyden:
    def __init__(self, lambda_init=0.01, lambda_factor=3.0):
        self.lam = lambda_init
        self.lambda_factor = lambda_factor
        self.J = None

    def step(self, network, x, y, recomputed = False):
        params = network.get_params()
        r = network.forward(x).ravel() - y.ravel()

        if self.J is None:
            self.J = self._jacobian(network, x)

        JtJ = self.J.T @ self.J
        Jtr = self.J.T @ r

        n_params = len(params)
        A = JtJ + self.lam * np.eye(n_params)
        delta = np.linalg.solve(A, -Jtr)

        new_params = params + delta
        network.set_params(new_params)

        r_new = network.forward(x).ravel() - y.ravel()
        old_loss = np.mean(r ** 2)
        new_loss = np.mean(r_new ** 2)

        if new_loss < old_loss:
            delta_r = r_new - r
            self.J = self.J + np.outer(delta_r - self.J @ delta, delta) / (delta @ delta)
            self.lam /= self.lambda_factor
        else:
            network.set_params(params)
            self.lam *= self.lambda_factor
            if not recomputed:
                self.J = self._jacobian(network, x)

    def _jacobian(self, network, x, eps=1e-5):
        params = network.get_params()
        n_data = x.shape[0]
        n_params = len(params)
        J = np.zeros((n_data, n_params))

        f0 = network.forward(x).ravel()

        for j in range(n_params):
            params_plus = params.copy()
            params_plus[j] += eps
            network.set_params(params_plus)
            f_plus = network.forward(x).ravel()
            J[:, j] = (f_plus - f0) / eps

        network.set_params(params)
        return J

    def optimize(self, network, x, y, epochs, jacobian_interval = 10):
        self.J = None
        history = []
        for epoch in range(epochs):
            recomputed = False
            if epoch % jacobian_interval == 0:
                self.J = self._jacobian(network, x)
                recomputed = True
            self.step(network, x, y, recomputed)
            history.append(network.loss(x, y))
        return history

    """
    FAILED TRY TO GET JACOBIAN ANALYTICALLY 
    
    def _jacobian_fast(self, network, x):
        params = network.get_params()
        n_data = x.shape[0]
        n_params = len(params)
        J = np.zeros((n_data, n_params))

        for i in range(n_data):
            xi = x[i:i+1]
            network.forward(xi)
            delta = np.ones((1, network.layers[-1].b.shape[0]))
            for layer in reversed(network.layers):
                delta = layer.backward(delta)

            J[i, :] = delta
        return J
        """


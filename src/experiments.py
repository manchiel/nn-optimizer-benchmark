import numpy as np
import time
from src.network import Network,Layer
from src.optimizers import GradientDescent, Adam, LevenbergMarquardt, LMBroyden, LBFGS

architectures = [
    [1, 4, 1],
    [1, 8, 1],
    [1, 16, 1],
    [1, 32, 1],
    [1, 64, 1]
]

n_data = [50, 100, 500, 1000, 5000]

def generate_data(n_samples, noise = 0.1, seed = 0):
    np.random.seed(seed)
    x = np.linspace(-np.pi, np.pi, n_samples).reshape(-1, 1)
    y = np.sin(x) + noise * np.random.randn(n_samples, 1)
    return x, y

def build_network(arhitecture):
    layers = []
    for i in range(len(arhitecture) - 1):
        if i == len(arhitecture) - 2:
            layers.append(Layer(arhitecture[i], arhitecture[i + 1], activation = 'linear'))
        else:
            layers.append(Layer(arhitecture[i], arhitecture[i+1]))
    return Network(layers)

def run_validation(epochs=200, arh=None, n_samples=50, seed=42):
    if arh is None:
        arh = [1, 8, 1]

    x, y = generate_data(n_samples)
    results = {}

    optimizers = {
        'GD': GradientDescent(),
        'Adam': Adam(),
        'LM': LevenbergMarquardt(),
        'LMB': LMBroyden(),
        'LBFGS': LBFGS()
    }

    np.random.seed(seed)
    base_net = build_network(arh)
    base_params = base_net.get_params().copy()

    for name, optimizer in optimizers.items():
        net = build_network(arh)
        net.set_params(base_params.copy())

        start = time.time()
        history = optimizer.optimize(net, x, y, epochs=epochs)
        elapsed = time.time() - start

        results[name] = {
            'history': history,
            'time': elapsed,
            'final_loss': net.loss(x, y)
        }

        print(f"{name:12} | loss: {net.loss(x, y):.6f} | time: {elapsed:.3f}")

    return results

def all_algos_on_all_nets():
    results = {}
    for architecture in architectures:
        results[str(architecture)] = run_validation(epochs=200, arh=architecture)
        print("-------------------------------------------")
    return results

def all_algos_on_all_datasets():
    results = {}
    for n_samples in n_data:
        print(f"\n====== Dataset size: {n_samples} samples ==========")
        results[n_samples] = {}
        for architecture in architectures:
            print(f"--- Architecture: {architecture} --------------")
            results[n_samples][str(architecture)] = run_validation(epochs= 500, arh=architecture, n_samples=n_samples)
            print("-------------------------------------------")
    return results

if __name__ == "__main__":
    print("----Experiment on small network----")
    results1 = run_validation()
    print("----Experiment on different networks----")
    results2 = all_algos_on_all_nets()
    print("----Experiment on different networks & datasets----")
    results3 = all_algos_on_all_datasets()


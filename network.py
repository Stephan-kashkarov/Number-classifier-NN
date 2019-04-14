import numpy as np
from layer import Layer

class Network:
    """
    Network class

    The network class is the full implementation of the
    neural network model

    Arguments:
        -> shape
        -> activation_f
        -> weights
    """

    def __init__(self, **kwargs):
        self.shape = kwargs.get("shape", [(3, 0), (3, 0), (3, 0)])
        self.activations = kwargs.get('activations', ['relu', 'sigmoid', 'softmax'])
        self.layers = []
        weights = kwargs.get('weights', None)
        for i, shpe in enumerate(self.shape):
            self.layers.append(
                Layer(
                    shape=shpe,
                    activation=self.activations[i],
                    weights=weights[i] if weights else None
                )
            )
        self.layers = np.array(self.layers)

    def execute(self):
        pass

    def error(self):
        pass

    def update(self):
        pass

    def train(self):
        pass

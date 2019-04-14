import numpy as np

activs = {
    'sigmoid': lambda x: x,
    'softmax': lambda x: x,
    'relu': lambda x: x,
}

class Neuron:
    def __init__(self, **kwargs):
        self.shape = (kwargs.get('x', 3), kwargs.get('y', 3))
        self.activ = activs.get(
            kwargs.get("activation_function", "sigmoid"),
            lambda x: x,
        )
        self.weights = np.random.rand(self.shape)
        self.activation = 0

    def execute(self, inputs):
        return self.activ(np.sum(np.multiply(self.weights, inputs)))



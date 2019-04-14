import numpy as np
from neuron import Neuron


activation_funcs = {
    'sigmoid': np.vectorize(lambda x: 1/(1+np.exp(x))),
    'relu': np.vectorize(lambda x: max(x, 0)),
    'softmax': np.vectorize(lambda x: np.exp(x)/np.sum(np.exp(x))),
}

run = np.vectorize(lambda x, y: x.execute(y))

class Layer:
    """
    Layer class

    This class contains a layer of neurons in a neural network
    it runs each neuron with a previous layer and applies an
    activation function to each

    The initaliser for this class takes the kwargs as follows

    Kwargs:
        -> x                   | The width of the input layer
        -> y                   | The height of the input layer
        -> activation          | The activation function to be mapped over the output
    """
    def __init__(self, **kwargs):
        self.shape = (kwargs.get("x", 3), kwargs.get('y', 1))
        self.activation = activation_funcs.get(kwargs.get('activation', 'sigmoid'))
        self.neurons = np.array(
            [
                Neuron(shape=self.shape)
                for x in range(
                    self.shape[0] * self.shape[1]
                )
            ]
        ).reshape(self.shape)
        self.output = np.empty(self.shape)

    def execute(self, inputs):
        self.output = self.activation(run(self.neurons, inputs).reshape(self.shape))

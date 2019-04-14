import numpy as np
from neuron import Neuron


activation_funcs = {
    'sigmoid': np.vectorize(lambda x: 1/(1+np.exp(x))),
    'relu': np.vectorize(lambda x: max(x, 0)),
    'softmax': np.vectorize(lambda x: np.exp(x)/np.sum(np.exp(x))),
}

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
        self.shape = (kwargs.get('shape', (3,)))
        self.activation = activation_funcs.get(kwargs.get('activation', 'sigmoid'))
        weights = kwargs.get('weights', None)
        self.neurons = np.array(
            list([
                Neuron(
                    shape=self.shape,
                    weights=weights.pop(0)
                )
                for x in range(
                    np.multiply(self.shape)
                )
            ])
        ).reshape(self.shape)
        self.output = np.empty(self.shape)

    def execute(self, inputs):
        """
        Execute method

        This method executes all neurons in layer and
        applys an activation function to each

        Arguments:
            -> inputs          | A Numpy array of the outputs of prev row
        """
        run = np.vectorize(lambda x: x.execute(inputs))
        self.output = self.activation(run(self.neurons).reshape(self.shape))

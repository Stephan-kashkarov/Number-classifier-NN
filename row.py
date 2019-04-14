import numpy as np
from neuron import Neuron

class Row:
    def __init__(self, **kwargs):
        self.shape = (kwargs.get("x", 3), kwargs.get('y', 1))
        self.softmax = True if kwargs.get('activ', None) == 'softmax' else False
        self.neurons = np.array(
            [
                Neuron(shape=self.shape, activ=kwargs.get('activ'))
                for x in range(
                    self.shape[0] * self.shape[1]
                )
            ]
        ).reshape(self.shape)

    def execute(self, inputs):
        pass

import numpy as np
from neuron import Neuron
from activation import (
	activation_funcs,
	der_funcs,
)


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
		self.shape = np.array(kwargs.get('shape', (3,)))
		self.prev_shape = kwargs.get('prev_shape', (3,))
		self.activation = activation_funcs.get(kwargs.get('activation', 'sigmoid'))
		weights = kwargs.get('weights', None)
		self.neurons = np.array(
			list([
				Neuron(
					shape=self.shape,
					weights=weights.pop(0) if weights else None
				)
				for x in range(
					np.prod(self.shape)
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
		self.output = np.array(list(self.activation(run(self.neurons).reshape(self.shape))))
		return self.output


class Input_layer:
	"""
	Input_layer

	This class acts an input layer for the program
	it takes a shape as an arguement and then contains all shapes
	within self.output
	"""

	def execute(self, image):
		"""
		Execute method

		This method allows of a network to dynamically
		extract image the same way as Layer class allowing
		for seamless execution

		Arguments:
			-> image          | An array of numbers used as input in shape self.shape
		"""
		return image

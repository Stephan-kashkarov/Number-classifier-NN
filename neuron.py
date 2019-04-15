import numpy as np

class Neuron:
	"""
	Neuron Initializer

	This method initalises an instance of a neuron
	Each neuron has the following internal varaibles

	Attributes:
		-> Shape               | This keeps track of the size of the inputs and weights
		-> Activ               | This is the function used to cap the range of the input between 1 and zero
		-> Weights             | This is a 2D array containing the weights of the system
		-> Activation          | This is the output of this neuron

	Kwargs:
		-> shape               | The shape of the input row
		-> weights             | The weights of system, if none provied will be random
		
	"""
	def __init__(self, **kwargs):
		self.shape = kwargs.get('shape', (3,))
		self.weights = kwargs.get('weights', np.random.rand(self.shape))
		self.activation = 0

	def execute(self, inputs):
		"""
		Execute method

		This method takes a numpy 2D array as an input and
		returns a single value in the form of an activation.

		Arguments:
			-> inputs              | The inputs given to the neuron network (Ideally of shape of self.shape)
		
		Returns:
			-> activation          | The output for the neuron
		"""
		self.activation = np.sum(np.multiply(self.weights, inputs))
		return self.activation



import numpy as np
from layer import Layer, Input_layer
from error import cross_entropy

class Network:
	"""
	Network class

	The network class is the full implementation of the
	neural network model

	Kwargs:
		-> shape               | an array containing the shape of each layer in the network
		-> activation_f        | an array containing each activation funtion in the network
		-> weights             | an array of arrays of the weight of each neuron to be preset
		-> output_lables       | an array containing the lables for each neuron in output row
	"""

	def __init__(self, **kwargs):
		self.shape = kwargs.get("shape", [(3,), (3,), (3,), (3,)])
		self.activations = kwargs.get('activations', ['relu', 'sigmoid', 'softmax'])
		self.layers = [Input_layer()]
		self.output_lables = kwargs.get('output_lables', ["1", "2", "3"])
		weights = kwargs.get('weights', None)
		for i, shpe in enumerate(self.shape[1:]):
			self.layers.append(
				Layer(
					shape=shpe,
					prev_shape=self.shape[i-1],
					activation=self.activations[i],
					weights=weights[i] if weights else None
				)
			)
		self.layers = np.array(self.layers)

	def execute(self, image, labels):
		"""
		Execute method

		This method takes an image and a label and runs the network
		accordingly. This will then print the confidence of each 
		layer.

		Arguments:
			-> image           | an array of numbers as input
			-> lable           | the intended confidence of each output
		"""
		outputs = np.array(image)
		for layer in self.layers:
			outputs = layer.execute(outputs)

		print("NN executed with these results:")
		for i, output in enumerate(outputs):
			print(f"{self.output_lables[i]}: {output} confidence")
		print(f"error: {cross_entropy(outputs, labels)}")


	def update(self):
		pass

	def train(self, images, lables):
		pass

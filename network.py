import numpy as np
from layer import Layer, Input_layer

class Network:
	"""
	Network class

	The network class is the full implementation of the
	neural network model

	Kwargs:
		-> shape              | an array containing the shape of each layer in the network
		-> activation_f       | an array containing each activation funtion in the network
		-> weights            | an array of arrays of the weight of each neuron to be preset
		-> output_lables      | an array containing the lables for each neuron in output row
	"""

	def __init__(self, **kwargs):
		self.shape = kwargs.get("shape", [(3,), (3,), (3,), (3,)])
		self.activations = kwargs.get('activations', ['relu', 'sigmoid', 'softmax'])
		self.layers = [Input_layer(shape=self.shape[0])]
		self.output_lables = kwargs.get('output_lables', None)
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

	def execute(self, image):
		outputs = np.array(image)
		for layer in enumerate(self.layers[1:], 0):
			outputs = layer.execute(outputs)

		print("NN executed with these results:")
		for i, output in enumerate(outputs):
			print(f"{self.output_lables[i]}: {output:2f} confidence")

	def error(self):
		pass

	def update(self):
		pass

	def train(self, images, lables):
		pass

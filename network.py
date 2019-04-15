import numpy as np
from layer import Layer, Input_layer
from error import cross_entropy, der_cross_entropy
from activation import der_funcs

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
		self.error = 1
		self.learning_rate = kwargs.get('learning_rate',0.01)

	def execute(self, image, label):
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
		
		self.error = cross_entropy(outputs, label)
		print(f"error: {self.error}")
		return outputs

	def backpropogate(self, outputs, labels):
		for index_layer, layer in enumerate(self.layers[:0:-1]):
			prev_layer = self.layers[index_layer - 1]
			for index_neuron, neuron in enumerate(layer.neurons):
				for index_weight, weight in enumerate(neuron.weights.flatten()):
					err_derivative = list(der_cross_entropy(outputs, labels))[index_neuron]
					activ_derivative = list(der_funcs[layer.activation](prev_layer.output))[index_neuron]
					weight_derivative = prev_layer.output[index_neuron]
					derivative = err_derivative * activ_derivative * weight_derivative
					new_weight = weight - (derivative*self.learning_rate)
					np.put(neuron.weights, index_weight, new_weight)



	def update(self):
		pass

	def train(self, images, labels):
		while images:
			image, label = images.pop, labels.pop
			self.execute(image, label)
			self.backpropogate()


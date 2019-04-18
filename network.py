import json
import numpy as np
from layer import Layer, Input_layer
from error import cross_entropy, der_cross_entropy
from activation import der_funcs, softmax

class Network:
	"""
	Network class

	The network class is the full implementation of the
	neural network model

	Kwargs:
		-> shape               | an array containing the shape of each layer in the network
		-> activation_f        | an array containing each activation funtion in the network
		-> weights             | an array of arrays of the weight of each neuron to be preset
		-> output_labels       | an array containing the lables for each neuron in output row
	"""

	def __init__(self, **kwargs):
		self.shape = kwargs.get("shape", [(3,), (3,), (3,), (3,)])
		self.activations = kwargs.get('activations', ['relu', 'sigmoid', 'softmax'])
		self.layers = [Input_layer()]
		self.output_labels = kwargs.get('output_labels', ["1", "2", "3"])
		weights = kwargs.get('weights', None)
		for i, shape in enumerate(self.shape[1:]):
			self.layers.append(
				Layer(
					shape=shape,
					prev_shape=self.shape[i],
					activation=self.activations[i],
					weights=weights[i] if weights else None
				)
			)
		self.layers = np.array(self.layers)
		self.error = 1
		self.learning_rate = kwargs.get('learning_rate',0.001)

	def execute(self, image, label, printing=False):
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

		output = list(outputs)
		if not printing:
			print(f"prediction: {self.output_labels[output.index(max(output))]}")
		if printing:
			print(f"Prediction: {self.output_labels[output.index(max(output))]}, Answer: {str(label)}")
		self.error = cross_entropy(outputs, label)
		print(f"error: {self.error}")
		if printing:
			return self.output_labels[output.index(max(output))] == str(label)
		return outputs

	def backpropogate(self, outputs, labels):
		"""
		backprpogate

		This is the training algorithm of the neural network.
		It operates in accordance to a number of mathematical formulas
		which i dont really understand

		TODO: Understand formulas

		The function takes some outputs and labels and then
		backpropogates through the network to find the ideal
		weight to give the correct answer.

		It then updates these weights by subtracting the change
		This change is found using derivatives.

		@param outputs | np.array | a list of outputs
		@param outputs | np.array | a list of labels for given outputs

		@returns error | Int      | The error in the system
		"""
		error = list(der_cross_entropy(outputs, labels))
		layers = self.layers[:0:-1]
		for index_layer, layer in enumerate(self.layers[:0:-1]):
			prev_layer = layers[index_layer + 1] if index_layer < len(layers) - 1 else self.layers[0]
			next_layer = layers[index_layer - 1] if index_layer > 0 else None
			layer.err_derivative = []
			layer.activ_derivative = []
			layer.weight_derivative = []
			for index_neuron, neuron in enumerate(layer.neurons):
				# neuron derivatives
				layer.activ_derivative.append(list(der_funcs[layer.activation](prev_layer.output))[index_neuron])
				if not next_layer:
					# print("a", index_layer, index_neuron)
					layer.err_derivative.append(error[index_neuron])
				else:
					# print("b", index_layer, index_neuron)
					temp = []
					for y in range(len(next_layer.neurons) - 1):
						# print(len(next_layer.neurons), y, index_layer)
						temp2 = next_layer.err_derivative[y] * next_layer.activ_derivative[y]
						for x in range(len(next_layer.neurons[0].weights)):
							temp.append(temp2*next_layer.neurons[y].weights[x])
					layer.err_derivative.append(sum(temp))
				for index_weight, weight in enumerate(neuron.weights):
					# weight derivative
					layer.weight_derivative.append(prev_layer.output[index_weight])
					# calculate total derivative
					derivative = layer.err_derivative[-1] * layer.activ_derivative[-1] * layer.weight_derivative[-1]
					# calculate and update new weight
					new_weight = weight + (derivative * self.learning_rate)
					neuron.weights[index_weight] = new_weight
		return error

	def train(self, images, labels):
		"""
		Train method

		This method takes a set of images and labels then feeds
		them to the neural network which then backpropogates with its outputs

		This then breaks when it reaches its minimum error and returns the
		weights used to acheive this error

		@param images | list | an array of images
		@param labels | list | an array of labels
		"""
		labels = list(labels)
		if not isinstance(labels[0], list):
			for i, x in enumerate(labels):
				a = [0 for x in range(10)]
				a[x] = 1
				labels[i] = a

		print("minimising images 2")
		for image in images:
			image = softmax(image)
		print("done")

		# n = 20
		while True:
			# if n > 0:
			# 	n -= 1
			for i in range(len(images)):
				image, label = images[i], labels[i]
			# error = self.error
				outputs = self.execute(image, label)
			# if self.error > error and n == 0:
				# print(f"Min error: {self.error}")
				# return self.save_weights()
				self.backpropogate(outputs, label)


	def save_weights(self):
		print("Saving...")
		output = {
			'error':self.error,
			'weights': []
		}
		for layer in self.layers[1:]:
			temp = []
			for neuron in layer.neurons:
				temp.append(list(neuron.weights))
			output['weights'].append(temp)
		
		with open(f"weights/weights_{self.error}.json", "w+") as w:
			json.dump(output, w)
		
		print(f"Saved at /weights/weights_{self.error}.json")

	def load_weights(self, filename):
		pass

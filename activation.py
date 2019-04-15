import numpy as np

def relu(arr):
	for val in arr:
		yield max(val, 0)

def der_relu(arr):
	for val in arr:
		yield 1 if val > 0 else 0


def sigmoid(arr):
	for val in arr:
		yield (1/(1+np.exp(-val)))

def der_sigmoid(arr):
	for val in arr:
		yield sigmoid(val) * sigmoid(1 - val)


def softmax(arr):
	base = sum([np.exp(x) for x in arr])
	for x in arr:
		yield np.exp(x)/base

def der_softmax(arr):
	for val in arr:
		yield 

activation_funcs = {
	'sigmoid': sigmoid,
	'relu': relu,
	'softmax': softmax,
}

der_funcs = {
	'sigmoid': der_sigmoid,
	'relu': der_relu,
	'softmax': der_softmax
}
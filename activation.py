import numpy as np

def relu(arr):
	for val in arr if isinstance(arr, list) else [arr]:
		yield max(val, 0)

def der_relu(arr):
	for val in arr if isinstance(arr, list) else [arr]:
		yield 1 if val > 0 else 0


def sigmoid(arr):
	for val in arr if isinstance(arr, list) else [arr]:
		yield (1/(1+np.exp(-val)))

def der_sigmoid(arr):
	for val in arr if isinstance(arr, list) else [arr]:
		yield np.multiply(list(sigmoid(val)), list(sigmoid(1 - val)))[0]


def softmax(arr):
	base = sum([np.exp(x) for x in arr])
	for x in arr if isinstance(arr, list) else [arr]:
		yield np.exp(x)/base

def der_softmax(arr):
	arr = arr if isinstance(arr, list) else [arr]
	base = sum([np.exp(val) for val in arr]) ** 2
	for val in arr:
		others = sum([np.exp(x) if x != val else 0 for x in arr])
		yield (np.exp(val) * others)/base

activation_funcs = {
	'sigmoid': sigmoid,
	'relu': relu,
	'softmax': softmax,
}

der_funcs = {
	sigmoid: der_sigmoid,
	relu: der_relu,
	softmax: der_softmax
}

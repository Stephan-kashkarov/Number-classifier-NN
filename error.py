import numpy as np

def cross_entropy(outputs, labels):
	arr = []

	for i, output in enumerate(outputs):
		try:
			a = labels[i] * np.log10(output)
		except:
			a = 0
		try:
			b = (1-labels[i]) * np.log10(1-output)
		except:
			b = 0
		arr.append(
			- (a + b)
		)
	return round(sum(arr), 3)

def der_cross_entropy(outputs, labels):
	for i, output in enumerate(outputs):
		yield -1 * (labels[i] * (1/output) + (1-labels[i]) * (1/(1-output)))


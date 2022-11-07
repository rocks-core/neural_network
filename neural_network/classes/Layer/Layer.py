import numpy as np
from neural_network.classes.ActivationFunctions import ActivationFunction


class Layer:
	def __init__(
			self,
			number_units: int,
			activation_function: ActivationFunction,
	) -> None:
		self.number_units = number_units
		self.activation_function = activation_function
		# internal variables needed for backprop
		self.outputs = None  # array of values
		self.nets = None  # array of values
		self.error_signals = None  # array of values

	def feedforward(self, input_vector: np.array, layer_weights: np.array) -> np.array:
		"""
		Given an input vector, performs the feedforwrd across the layer, automatically updating the
		layer nets and outputs field (a.k.a. the layer internal state)

		:param input_vector: array of numbers
		:param layer_weights: array of array, i-th element is the array containing the entering weights of i-th unit of the layer next to this one
		:return: the array containing the outputs of the units of this layer
		"""
		input_vector = np.insert(input_vector, 0, 1)  # adding bias term to input

		# for each unit compute dot product
		self.nets = np.array([
			np.dot(unit_weights, input_vector)
			for unit_weights in layer_weights
		])

		# for each net result apply activation function
		self.outputs = self.activation_function.f(self.nets)

		return self.outputs  # note this function does not return the bias term!

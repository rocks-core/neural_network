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
		self.weights = None  # np matrix
		self.previous_layer = None
		self.next_layer = None
		self.built = False

	def build(self, previous_layer):
		self.weights = np.random.rand(self.number_units, previous_layer.number_units + 1)
		previous_layer.next_layer = self
		self.previous_layer = previous_layer
		self.built = True

	def feedforward(self, input_vector: np.array) -> np.array:
		"""
		Given an input vector, performs the feedforwrd across the layer, automatically updating the
		layer nets and outputs field (a.k.a. the layer internal state)

		:param input_vector: np array of numbers with shape (n, 1)
		:param layer_weights: matrices of weights,
		:return: the array containing the outputs of the units of this layer
		"""
		if not self.built:
			raise Exception("Layer not built, add it to a network")

		input_vector = np.insert(input_vector, 0, 1, axis=0)  # adding bias term to input

		# compute input times matrix weights
		self.nets = np.dot(self.weights, input_vector)

		# for each net result apply activation function
		self.outputs = self.activation_function.f(self.nets)

		return self.outputs  # note this function does not return the bias term!

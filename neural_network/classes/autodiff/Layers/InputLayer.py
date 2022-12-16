import numpy as np

from . import Layer
from neural_network.classes.autodiff.ActivationFunctions import ActivationFunction
from neural_network.classes.Initializer import Initializer
from neural_network.classes.autodiff import AutodiffFramework


class InputLayer(Layer):

	def __init__(
			self,
			input_shape: tuple,
			number_units: int,
			activation_function: ActivationFunction,
			initializer: Initializer
	) -> None:
		super().__init__(number_units, activation_function, initializer)
		self.input_shape = input_shape

	def build(self, af: AutodiffFramework) -> None:
		self.af = af
		self.weights = af.add_variable(self.initializer(shape=(self.input_shape[-1] + 1, self.number_units)))
		self.built = True

	def match_shape(self, v):
		# check if shape of v match the input_shape provided
		# if a None value is present in the input_shape provided any value is admitted for that axis in the shape of v
		if len(self.input_shape) != len(v.shape):
			return False
		for s1, s2 in zip(self.input_shape, v.shape):
			if s1 is None:
				continue
			if s1 != s2:
				return False
		return True

	def feedforward(self, inputs):
		if not self.built:
			raise Exception("Layers not built, add it to a network")
		if not self.match_shape(inputs):
			raise Exception("input shape not matching the shape provided")

		inputs = self.af.concat(np.ones((inputs.shape[0], 1)), inputs, ax=1)
		net = self.af.matmul(inputs, self.weights)
		out = self.activation_function.f(self.af, net)
		return out

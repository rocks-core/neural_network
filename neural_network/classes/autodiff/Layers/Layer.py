from neural_network.classes.autodiff import AutodiffFramework
from neural_network.classes.autodiff.ActivationFunctions import ActivationFunction
from neural_network.classes.Initializer import Initializer
from abc import abstractmethod

class Layer:
	def __init__(
			self,
			number_units: int,
			activation_function: ActivationFunction,
			initializer: Initializer,
	) -> None:
		self.number_units = number_units
		self.activation_function = activation_function
		self.initializer = initializer
		# internal variables needed for backprop
		self.outputs = None  # array of values
		self.weights = None  # np matrix
		self.bias = None
		self.previous_layer = None
		self.next_layer = None
		self.built = False
		self.af = None

	def build(self, af: AutodiffFramework, previous_layer):
		self.af = af
		previous_layer.next_layer = self
		self.previous_layer = previous_layer
		self.built = True

	@abstractmethod
	def feedforward(self, inputs):
		pass

	def __call__(self, inputs, *args, **kwargs):
		return self.feedforward(inputs)

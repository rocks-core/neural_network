from . import Layer
from neural_network.classes.autodiff import AutodiffFramework
import numpy as np


class DenseLayer(Layer):
	def build(self, af: AutodiffFramework, previous_layer):
		super().build(af, previous_layer)
		self.weights = af.add_variable(self.initializer(shape=(previous_layer.number_units + 1, self.number_units)))
		# self.bias = af.add_variable(self.initializer(shape=(1, self.number_units)))

	def feedforward(self, inputs):
		if not self.built:
			raise Exception("Layers not built, add it to a network")

		inputs = self.af.concat(np.ones((inputs.shape[0], 1)), inputs, ax=1)
		net = self.af.matmul(inputs, self.weights)
		out = self.activation_function.f(self.af, net)
		return out

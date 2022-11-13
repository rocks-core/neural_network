import numpy as np
from neural_network.classes.Layer import Layer
from neural_network.classes.ActivationFunctions import ActivationFunction


class HiddenLayer(Layer):
	def __init__(
			self,
			number_units: int,
			activation_function: ActivationFunction
	) -> None:
		super().__init__(number_units, activation_function)

	def backpropagate(
			self
	) -> np.array:
		"""
		Backpropagates the error signals from the next layer, generating the error signals of the current
		layer (and updating the internal state) and computing the deltas of the current layer incoming weights

		:param next_layer_error_signals: array of numbers, contains the error signals of the layer next the current one;
			the i-th element of this array corresponds to the i-th error signal of the i-th unit of the next layer
		:param next_layer_weights: array of arrays, it's a matrix (even if the data type is a vector) representing the
			weights between layer1 (the current layer) and layer2 (the next layer); the i-th row
			correponds to the i-th unit (of the layer 2) incoming weights, while the j-th column corresponds to the
			weights of the j-th unit (of the layer1) edges.
		:param previous_layer_outputs: array of numbers, contains the outputs of the previous layer units
		:return: an array of arrays, containing the deltas to update the current layer weight; in particular, the
			i-th row corresponds to the deltas of the i-th unit incoming weight.
		"""
		previous_layer_outputs = np.insert(self.previous_layer.outputs, 0, 1, axis=0)

		# layer_weights[i,j] is the weight that j-th neuron assign to the i-th input (the opposite of what seen at lesson)
		# remove the weight corresponding to bias
		next_layer_weights = self.next_layer.weights[:, 1:]

		# for each next layer note do dot product between error signal and incoming weight from current unit
		self.error_signals = np.dot(next_layer_weights.T, self.next_layer.error_signals)

		derivative_applied_on_nets = self.activation_function.derivative_f(self.nets)
		self.error_signals = np.multiply(self.error_signals, derivative_applied_on_nets)

		return np.dot(self.error_signals, previous_layer_outputs.T)

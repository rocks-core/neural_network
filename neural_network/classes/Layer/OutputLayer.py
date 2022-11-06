import numpy as np
from neural_network.classes.Functions.ActivationFunctions import ActivationFunction
from neural_network.classes.Layer import Layer

class OutputLayer(Layer):
	def __init__(
			self,
			number_units: int,
			activation_function: ActivationFunction
	) -> None:
		super().__init__(number_units, activation_function)

	def backpropagate(
			self,
			expected_output: np.array,
			previous_layer_outputs: np.array
	) -> np.array:
		"""
		Backpropagates the error signals from the next layer, generating the error signals of the current
		layer (and updating the internal state) and computing the deltas of the current layer incoming weights

		:param expected_output: array of numbers, contains the outputs expected from the current layer
			(after a forwarding phase done on the inputs)
		:param previous_layer_outputs: array of numbers, contains the outputs of the previous layer units
		:return: an array of arrays, containing the deltas to update the current layer weight; in particular, the
			i-th row corresponds to the deltas of the i-th unit incoming weight.
		"""
		# adding bias to previous layer output
		previous_layer_outputs = np.insert(previous_layer_outputs, 0, 1, axis=1)

		# for each node compute difference between output and multiply for derivative of net
		output_difference = np.subtract(expected_output, self.outputs)

		self.error_signals = np.multiply(
			output_difference,
			np.array(
				list(map(self.activation_function.derivative_f, self.nets))
			)
		)

		# compute delta
		return np.dot(self.error_signals.T, previous_layer_outputs)

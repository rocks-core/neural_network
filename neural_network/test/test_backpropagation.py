import unittest
import numpy as np
from neural_network.classes.Layers import OutputLayer, HiddenLayer
from neural_network.classes.LossFunctions import MSE
from neural_network.classes.ActivationFunctions import Linear


class BackpropagationTests(unittest.TestCase):
	def test_output_layer(self):
		layer = OutputLayer(
			number_units=1,
			activation_function=Linear(),
			loss_function=MSE()
		)
		layer.nets = np.array([[1.04]])
		layer.outputs = np.array([[1.04]])

		# simulating backpropagation values
		expected_output = np.array([[0]])
		previous_layer_outputs = np.array([[2.4], [1.8]])

		deltas = layer.backpropagate(
			expected_output=expected_output,
			previous_layer_outputs=previous_layer_outputs
		)

		expected_error_signals = np.array([[-1.04]])
		expected_deltas = np.array([[-1.04], [-2.496], [-1.872]])

		# testing error signals
		for expected_unit_error_signals in expected_error_signals:
			self.assertListEqual(list(expected_unit_error_signals), list(layer.error_signals))

		for (expected_unit_deltas, real_unit_deltas) in zip(expected_deltas, deltas):
			self.assertListEqual(list(expected_unit_deltas), list(real_unit_deltas))

	def test_hidden_layer(self):
		layer = HiddenLayer(
			number_units=2,
			activation_function=Linear()
		)
		layer.outputs = np.array([[1], [2.4], [1.8]])
		layer.nets = np.array([[2.4], [1.8]])

		next_layer_error_signals = np.array([[-1.04]])
		next_layer_weights = np.array([[0.2, 0.2, 0.2]])
		previous_layer_outputs = np.array([[2], [3]])

		deltas = layer.backpropagate(
			next_layer_error_signals=next_layer_error_signals,
			next_layer_weights=next_layer_weights,
			previous_layer_outputs=previous_layer_outputs
		)
		deltas = list(deltas)
		expected_deltas = [
			[-0.208, -0.416, -0.624],
			[-0.208, -0.416, -0.624],
			[-0.208, -0.416, -0.624]
		]
		for (l1, l2) in zip(deltas, expected_deltas):
			for e1, e2 in zip(l1, l2):
				self.assertAlmostEqual(e1, e2, places=3)

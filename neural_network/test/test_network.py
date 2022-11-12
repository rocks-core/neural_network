import numpy as np
import unittest
from neural_network.classes import MLClassifier, ActivationFunctions


class NetworkTests(unittest.TestCase):
	def test_1(self):
		number_inputs = 2
		layer_0_units = 2
		layer_1_units = 1

		classifier = MLClassifier(number_inputs=number_inputs, layer_sizes=(layer_0_units, layer_1_units),
								  activation_functions=(
									  ActivationFunctions.Linear(),
									  ActivationFunctions.Linear()
								  ), n_epochs=1)

		inputs = np.array([2, 3]).reshape(-1, 1)
		output = np.array([[0]])

		classifier.layers[0]["weights"] = np.array([
			[0.4, 0.4, 0.4],
			[0.3, 0.3, 0.3]
		])
		classifier.layers[1]["weights"] = np.array([
			[0.2, 0.2, 0.2]
		])

		self.assertEqual(classifier.predict(inputs)[0][0], 1.04)
		classifier.fit([inputs], [output])

		self.assertEqual(
			len(classifier.layers[1]["layer"].error_signals),
			1
		)
		self.assertEqual(
			classifier.layers[1]["layer"].error_signals[0][0],
			-1.04
		)

		self.assertEqual(
			len(classifier.layers[0]["weights"]),
			2
		)
		self.assertListEqual(list(classifier.layers[0]["weights"][0]), [0.3792, 0.3584, 0.3376])
		self.assertListEqual(list(classifier.layers[0]["weights"][1]), [0.2792, 0.2584, 0.2376])

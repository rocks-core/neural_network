from neural_network.classes.Layer import HiddenLayer, OutputLayer
from neural_network import utils
import numpy as np


__all__ = ["MLClassifier"]


class MLClassifier:
	def __init__(
			self,
			number_inputs: int,
			layer_sizes: tuple,
			activation_functions: tuple,
			derivative_loss_function,
			alpha: float = 0.0001,
			regularization_term: float = 0.0001,
			batch_size: int = 100,
			learning_rate: float = 0.1,
			n_epochs: int = 100,
			shuffle: bool = False,
			verbose: bool = False,
			momentum: float = 0.9,
			nesterovs_momentum: bool = False
	):
		self.number_inputs = number_inputs
		self.number_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.activation_functions = activation_functions
		self.alpha = alpha
		self.regularization_term = regularization_term
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.shuffle = shuffle
		self.verbose = verbose
		self.momentum = momentum
		self.nesterovs_momentum = nesterovs_momentum

		# creating a structure to hold the layer related informations
		self.layers = {
			layer_index: {
				"layer":
					HiddenLayer(layer_units, layer_activation_fun)
					if layer_index != (self.number_layers - 1)
					else OutputLayer(layer_units, layer_activation_fun, derivative_loss_function),
				"weights":
					np.random.rand(layer_units, layer_sizes[layer_index - 1] + 1) * 0.5 - 0.2
					if layer_index != 0 else np.random.rand(self.layer_sizes[0], (number_inputs + 1)) * 0.5 - 0.2
			}
			for (layer_index, (layer_units, layer_activation_fun)) in enumerate(zip(layer_sizes, activation_functions))
		}

	def __update_weights(self, deltas: list):
		"""
		Update the weights of the network layers

		:param deltas: list of arrays, i-th element corresponds to the array of deltas of the i-th layer
		"""
		# iterate over the layers
		for ((_, layer), layer_deltas) in zip(self.layers.items(), deltas):
			# computing new weights for the layer
			layer["weights"] = np.array([
				unit_old_weights + self.learning_rate * delta - 2 * self.regularization_term * unit_old_weights
				for unit_old_weights, delta in zip(layer["weights"], layer_deltas)
			])

	def __fit_pattern(self, pattern: np.array, expected_output: np.array) -> list:
		"""
		Fits the neural network using the single specified pattern

		:param pattern: array of numbers, input features
		:param expected_output: array of number, expected outputs for this pattern
		:return: list of array of arrays, a list containing for each layer the deltas of its weights; the list
		is ordered, so the i-th element contains the deltas for the weights of the i-th layer
		"""
		deltas = []

		# forwarding phase
		self.predict(pattern)

		# backpropagation phase
		hidden_layers_indices = [_ for _ in range(self.number_layers-1)]
		output_layer = self.layers[self.number_layers-1]

		# output layer
		output_layer_deltas = output_layer["layer"].backpropagate(
			expected_output,
			self.layers[hidden_layers_indices[-1]]["layer"].outputs
		)
		deltas.insert(0, output_layer_deltas)

		# hidden layers
		hidden_layers_indices.reverse()
		backpropagating_layer = output_layer["layer"]

		for hidden_layer_index in hidden_layers_indices:
			hidden_layer = self.layers[hidden_layer_index]["layer"]
			backpropagating_layer_weights = self.layers[hidden_layer_index+1]["weights"]

			previous_layer_outputs = self.layers[hidden_layer_index-1]["layer"].outputs if hidden_layer_index != 0 else pattern

			hidden_layer_deltas = hidden_layer.backpropagate(
				backpropagating_layer.error_signals,
				backpropagating_layer_weights,
				previous_layer_outputs
			)
			deltas.insert(0, hidden_layer_deltas)

			# update reference to layer which backpropagates
			backpropagating_layer = hidden_layer

		return deltas

	def fit(self, inputs: np.array, expected_outputs: np.array):
		"""
		:param inputs:
		:param expected_outputs:
		:return:
		"""
		for iter_number in range(self.n_epochs):  # iterating for the specified epochs
			if self.verbose:
				print(f"Iteration {iter_number+1}/{self.n_epochs}")
			batched_patterns = [_ for _ in zip(inputs, expected_outputs)]  # group patterns in batches
			for (batch_number, batch) in enumerate(utils.chunks(batched_patterns, self.batch_size)):  # iterate over batches
				sum_of_deltas = [] # accumulator of deltas belonging to the batch
				for (pattern, expected_output) in batch:  # iterate over pattern of a single batch
					deltas = self.__fit_pattern(pattern, expected_output)

					# accumulate deltas of the same batch
					if sum_of_deltas == []:
						sum_of_deltas = deltas
					else:
						for index in range(len(sum_of_deltas)):
							sum_of_deltas[index] += deltas[index]

				# at the end of batch update weights
				self.__update_weights(deltas)

	def predict(self, input: np.array) -> np.array:
		"""
		Given an input vectors, feedforwards over all the layers
		:param input:
		:return: the network output
		"""
		layer_input = input
		for (_, layer_info) in self.layers.items():
			layer = layer_info["layer"]
			weights = layer_info["weights"]

			output = layer.feedforward(layer_input, weights)
			layer_input = output
		return output

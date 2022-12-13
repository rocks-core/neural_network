from neural_network.classes.LossFunctions import LossFunction
from neural_network import utils
from neural_network.classes.Results import Result
from neural_network.classes.Metrics import MetricConverter
import numpy as np
import pickle

__all__ = ["Model"]


class Model:
	def __init__(
			self,
			layers: list,
			loss: LossFunction,
			optimizer,
			metrics: list,
			shuffle: bool = False,
			verbose: bool = False,
	):
		self.layers = []
		self.build_layer(layers, loss)
		self.loss = loss
		self.number_layers = len(layers)
		self.optimizer = optimizer

		self.metrics = []
		for m in metrics:
			if isinstance(m, str):
				self.metrics.append(MetricConverter.get_from_string(m))
			else:
				self.metrics.append(m)
		self.metrics_score = {}
		self.metrics_history = {}

		self.shuffle = shuffle
		self.verbose = verbose
		self.early_stop = False

	def build_layer(self, layers, loss):
		layers[0].build()
		self.layers.append(layers[0])
		for layer in layers[1:-1]:
			layer.build(self.layers[-1])
			self.layers.append(layer)
		layers[-1].build(self.layers[-1], loss)
		self.layers.append(layers[-1])

	def fit_pattern(self, pattern: np.array, expected_output: np.array) -> list:
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

		reversed_layer = list(reversed(self.layers))
		output_layer = reversed_layer.pop(0)
		output_layer_deltas = output_layer.backpropagate(expected_output)
		deltas.insert(0, output_layer_deltas)

		for layer in reversed_layer:
			hidden_layer_deltas = layer.backpropagate()
			deltas.insert(0, hidden_layer_deltas)

		return deltas

	def initialize_metrics(self, val: bool):
		for m in self.metrics:
			self.metrics_history[m.name] = []
			if val:
				self.metrics_history["val_" + m.name] = []

	def update_metrics(self, output, expected_output, val_output, val_expected_output):
		for m in self.metrics:
			s = m.f(expected_output, output)
			self.metrics_score[m.name] = s
			self.metrics_history[m.name].append(s)
			if val_output is not None and val_expected_output is not None:
				s = m.f(val_expected_output, val_output)
				self.metrics_score["val_" + m.name] = s
				self.metrics_history["val_" + m.name].append(s)

	def print_metrics(self):
		for k, v in self.metrics_score.items():
			print(f"{k}: {v:.4f}", end="\t")
		print()

	def fit(
			self,
			inputs: np.array,
			expected_outputs: np.array,
			validation_data: list = None,
			epochs: int = 100,
			batch_size:int = 100,
			callbacks: list = []
	) -> Result:
		"""
		:param inputs:
		:param expected_outputs:
		:param validation_data:
		:param epochs:
		:param callbacks:
		:return:
		"""

		utils.check_input_shape(inputs)  # if input shape is (n,) transform it to row vector (shape (1, n))
		utils.check_output_shape(expected_outputs)  # if output shape is (n,) transform label to column vector (shape (n, 1))
		if validation_data:
			utils.check_input_shape(validation_data[0])
			utils.check_output_shape(validation_data[1])

		self.initialize_metrics(bool(validation_data))

		# iterating over the epochs
		for iter_number in range(epochs):
			# compute output to compute the metrics score
			output = self.predict(inputs)
			val_output = None
			val_expected_output = None
			# if a validation set is specified, compute output also for it over it
			if validation_data:
				val_output = self.predict(validation_data[0])
				val_expected_output = validation_data[1]
			# update the scores of the metrics
			self.update_metrics(output, expected_outputs, val_output, val_expected_output)
			if self.verbose:
				print("Epoch ", iter_number + 1, end=" ")
				self.print_metrics()

			for callback in callbacks:
				callback(self)

			if self.early_stop:
				break

			# group patterns in batches
			batches = utils.chunks(inputs, expected_outputs, batch_size)
			for (batch_in, batch_out) in batches:  # iterate over batches
				self.optimizer.apply(self, batch_in, batch_out)  # update the weights

				# at the end of batch update weights

		return Result(metrics=self.metrics_score, history=self.metrics_history)

	def predict(self, input: np.array) -> np.array:
		"""
		Given an input vectors, feedforwards over all the layers
		:param input:
		:return: the network output
		"""

		utils.check_input_shape(input)
		layer_input = input
		for layer in self.layers:
			output = layer.feedforward(layer_input)
			layer_input = output
		return output

	def evaluate(self, input: np.array, expected_output: np.array):
		utils.check_input_shape(input)
		utils.check_output_shape(expected_output)

		output = self.predict(input)
		metrics_score = {}
		for m in self.metrics:
			metrics_score[m.name] = m.f(output, expected_output)
		return metrics_score

	def evaluate_result(self, input: np.array, expected_output: np.array):
		utils.check_input_shape(input)
		utils.check_output_shape(expected_output)

		output = self.predict(input)
		metrics_score = {}
		for m in self.metrics:
			metrics_score[m.name] = m.f(output, expected_output)
		result = Result(metrics=metrics_score, history={})
		return result

	def get_weights(self):
		weights = []
		for layer in self.layers:
			weights.append(layer.weights.copy())
		return weights

	def set_weights(self, weights):
		for layer, w in zip(self.layers, weights):
			layer.weights = w.copy()

	def dump_model(self, path):
		with open(path, "wb") as file:
			pickle.dump(self, file)

	@staticmethod
	def load_model(path):
		with open(path, "rb") as file:
			return pickle.load(file)

	def dump_weights(self, path):
		with open(path, "wb") as file:
			pickle.dump(self.get_weights(), file)

	def load_weights(self, path):
		with open(path, "rb") as file:
			self.set_weights(pickle.load(file))

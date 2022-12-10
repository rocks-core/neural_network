from neural_network.classes.LossFunctions import LossFunction
from neural_network import utils
from neural_network.classes.Results import Result
from neural_network.classes.Metrics import MetricConverter
import numpy as np
import pickle
import wandb

__all__ = ["MLClassifier"]


class MLClassifier:
	def __init__(
			self,
			layers: list,
			loss: LossFunction,
			optimizer,
			metrics: list,
			batch_size: int = 100,
			n_epochs: int = 100,
			shuffle: bool = False,
			verbose: bool = False,
			log_wandb: bool = False
	):

		layers[0].build()
		self.layers = [layers[0]]
		for layer in layers[1:-1]:
			layer.build(self.layers[-1])
			self.layers.append(layer)
		layers[-1].build(self.layers[-1], loss)
		self.layers.append(layers[-1])

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

		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.shuffle = shuffle
		self.verbose = verbose
		self.log_wandb = log_wandb

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
			s = m.f(output, expected_output)
			self.metrics_score[m.name] = s
			self.metrics_history[m.name].append(s)
			if val_output is not None and val_expected_output is not None:
				s = m.f(val_output, val_expected_output)
				self.metrics_score["val_" + m.name] = s
				self.metrics_history["val_" + m.name].append(s)

	def print_metrics(self):
		for k, v in self.metrics_score.items():
			print(k, str(v), end="\t")
		print()

	def fit(self, inputs: np.array, expected_outputs: np.array, validation_data: list = None) -> Result:
		"""
		:param inputs:
		:param expected_outputs:
		:param validation_data:
		:return:
		"""

		if len(inputs.shape) == 1:
			inputs = inputs.reshape(1, -1)  # transform input pattern to raw vector (shape (1, n))
		if len(expected_outputs.shape) == 1:  # transform label to column vector (shape (n, 1))
			expected_outputs = expected_outputs.reshape(-1, 1)

		if validation_data:
			if len(validation_data[0].shape) == 1:
				validation_data[0] = validation_data[0].reshape(-1, 1)
			if len(validation_data[1].shape) == 1:
				validation_data[1] = validation_data[1].reshape(-1, 1)

		self.initialize_metrics(bool(validation_data))
		for iter_number in range(self.n_epochs):  # iterating for the specified epochs
			output = self.predict(inputs)
			val_output = None
			val_expected_output = None
			if validation_data:
				val_output = self.predict(validation_data[0])
				val_expected_output = validation_data[1]
			self.update_metrics(output, expected_outputs, val_output, val_expected_output)

			if self.log_wandb:
				wandb.log(self.metrics_score)
			if self.verbose:
				print("Epoch ", iter_number+1, end=" ")
				self.print_metrics()
			# group patterns in batches
			for (batch_number, (batch_in, batch_out)) in enumerate(
					utils.chunks(inputs, expected_outputs, self.batch_size)):  # iterate over batches

				# deltas = self.__fit_pattern(batch_in, batch_out)
				# deltas = list(map(lambda x: np.divide(x, len(batch_out)), deltas))

				# at the end of batch update weights
				self.optimizer.apply(self, batch_in, batch_out)  # changed from delta to sum_of_deltas
				# the optimizer calling the fit_pattern function will update the metrics

		return Result(metrics=self.metrics_score, history=self.metrics_history)

	def predict(self, input: np.array) -> np.array:
		"""
		Given an input vectors, feedforwards over all the layers
		:param input:
		:return: the network output
		"""

		if len(input.shape) == 1:
			input = input.reshape(1, -1)
		layer_input = input
		for layer in self.layers:
			output = layer.feedforward(layer_input)
			layer_input = output
		return output

	def evaluate(self, input: np.array, expected_output: np.array):
		if len(expected_output.shape) == 1:  # transform label to column vector (shape (n, 1))
			expected_output = expected_output.reshape(-1, 1)
		output = self.predict(input)
		metrics_score = {}
		for m in self.metrics:
			metrics_score[m.name] = m.f(output, expected_output)
		return metrics_score

	def evaluate_result(self, input: np.array, expected_output: np.array):
		if len(expected_output.shape) == 1:  # transform label to column vector (shape (n, 1))
			expected_output = expected_output.reshape(-1, 1)
		output = self.predict(input)
		metrics_score = {}
		for m in self.metrics:
			metrics_score[m.name] = m.f(output, expected_output)
		result = Result(metrics=metrics_score, history={})
		return result

	def dump_model(self, path):
		with open(path, "wb") as file:
			pickle.dump(self, file)

	@staticmethod
	def load_model(path):
		with open(path, "rb") as file:
			return pickle.load(file)

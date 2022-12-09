from neural_network.classes.LossFunctions import LossFunction
from neural_network import utils
from neural_network.classes.Validation import EarlyStopping
from neural_network.classes.Results import Result
import numpy as np
import pickle

__all__ = ["MLClassifier"]


class MLClassifier:
	def __init__(
			self,
			layers: list,
			loss: LossFunction,
			optimizer,
			batch_size: int = 100,
			n_epochs: int = 100,
			shuffle: bool = False,
			verbose: bool = False,
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
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.shuffle = shuffle
		self.verbose = verbose

	def fit_pattern(self, pattern: np.array, expected_output: np.array) -> list:
		"""
		Fits the neural network using the single specified pattern

		:param pattern: array of numbers, input features
		:param expected_output: array of number, expected outputs for this pattern
		:return: list of array of arrays, a list containing for each layer the deltas of its weights; the list
		is ordered, so the i-th element contains the deltas for the weights of the i-th layer
		"""
		deltas = []

		if len(pattern.shape) == 1:
			pattern = pattern.reshape(1, -1)  # transform input pattern to raw vector (shape (n, 1))

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

	def fit(
			self,
			inputs: np.array,
			expected_outputs: np.array,
			validation_data: list = None,
			early_stopping: EarlyStopping = None
	) -> Result:
		"""
		:param inputs:
		:param expected_outputs:
		:param validation_data:
		:param early_stopping:
		:return:
		"""
		train_loss = []
		train_accuracy = []
		validation_loss = []
		validation_accuracy = []

		if len(expected_outputs.shape) == 1:
			expected_outputs = expected_outputs.reshape(-1, 1)

		if validation_data and len(validation_data[1].shape) == 1:
			validation_data[1] = validation_data[1].reshape(-1, 1)

		# iterating over the epochs
		for iter_number in range(self.n_epochs):
			# append computed loss/accuracy over training set
			train_loss.append(np.mean(self.loss.f(expected_outputs, self.predict(inputs))))
			train_accuracy.append(self.evaluate(inputs, expected_outputs))  # predict all the inputs together

			# if a validation set is specified, compute loss/accuracy over it
			if validation_data:
				validation_loss.append(np.mean(self.loss.f(validation_data[1], self.predict(validation_data[0]))))
				validation_accuracy.append(self.evaluate(validation_data[0], validation_data[1]))

			# verify if it's the case on an early stopping
			if early_stopping is not None:
				if early_stopping.add_monitored_value(
					self.layers,
					{
						"train_loss": train_loss[-1],
						"tran_accuracy": train_accuracy[-1],
						"validation_loss": validation_loss[-1],
						"validation_accuracy": validation_accuracy[-1]
					}
				): # check if it is a case of early stopping
					# restore best weights if the end user specified so
					self.layers = early_stopping.get_best_weights()
					print("EARLY STOPPING")
					break # stop training

			if self.verbose:
				print(
					f"Iteration {iter_number + 1}/{self.n_epochs}\tLoss {train_loss[-1]:.5f}\tAccuracy {train_accuracy[-1]:.5f}",
					end="")
				if validation_data:
					print(f"\tval loss {validation_loss[-1]:.5f}\tval accuracy {validation_accuracy[-1]:.5f}", end="")
				print("")

			# group patterns in batches
			batches = utils.chunks(inputs, expected_outputs, self.batch_size)
			for (batch_in, batch_out) in batches:  # iterate over batches
				self.optimizer.apply(self, batch_in, batch_out) # update the weights

		result = Result(
			metrics={
				"train_loss": train_loss[-1],
				 "train_acc": train_accuracy[-1],
				 "val_loss": validation_loss[-1],
				 "val_acc": validation_accuracy[-1]
			},
			result={
				"train_loss_curve": train_loss,
				"train_acc_curve": train_accuracy,
				"val_loss_curve": validation_loss,
				"val_acc_curve": validation_accuracy
			}
		)
		return result

	def predict(self, input: np.array) -> np.array:
		"""
		Given an input vectors, feedforwards over all the layers
		:param input:
		:return: the network output
		"""
		layer_input = input
		for layer in self.layers:
			output = layer.feedforward(layer_input)
			layer_input = output
		return output

	def evaluate(self, input: np.array, expected_output: np.array):
		output = self.predict(input)
		return np.mean(expected_output == np.rint(output))

	def dump_model(self, path):
		with open(path, "wb") as file:
			pickle.dump(self, file)

	@staticmethod
	def load_model(path):
		with open(path, "rb") as file:
			return pickle.load(file)

from neural_network.classes.Layer import HiddenLayer, OutputLayer, InputLayer
from neural_network import utils
import numpy as np

__all__ = ["MLClassifier"]


class MLClassifier:
    def __init__(
            self,
            input_shape: tuple,
            layers: list,
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
        self.layers = [InputLayer(input_shape)]
        for layer in layers:
            layer.build(self.layers[-1])
            self.layers.append(layer)
        self.number_layers = len(layers)
        self.alpha = alpha
        self.regularization_term = regularization_term
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.verbose = verbose
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum

    def __update_weights(self, deltas: list):
        """
		Update the weights of the network layers

		:param deltas: list of arrays, i-th element corresponds to the array of deltas of the i-th layer
		"""
        # iterate over the layers
        for layer, delta in zip(self.layers[1:], deltas):
            layer.weights = layer.weights + self.learning_rate * delta - 2 * self.regularization_term * layer.weights

    def __fit_pattern(self, pattern: np.array, expected_output: np.array) -> list:
        """
		Fits the neural network using the single specified pattern

		:param pattern: array of numbers, input features
		:param expected_output: array of number, expected outputs for this pattern
		:return: list of array of arrays, a list containing for each layer the deltas of its weights; the list
		is ordered, so the i-th element contains the deltas for the weights of the i-th layer
		"""
        deltas = []

        if len(pattern.shape) == 1:
            pattern = pattern.reshape(-1, 1)  # transform input pattern to raw vector (shape (n, 1))

        # forwarding phase
        self.predict(pattern)

        reversed_layer = list(reversed(self.layers))
        output_layer = reversed_layer.pop(0)
        output_layer_deltas = output_layer.backpropagate(expected_output)
        deltas.insert(0, output_layer_deltas)

        for layer in reversed_layer[:-1]:
            hidden_layer_deltas = layer.backpropagate()
            deltas.insert(0, hidden_layer_deltas)

        return deltas

    def fit(self, inputs: np.array, expected_outputs: np.array):
        """
		:param inputs:
		:param expected_outputs:
		:return:
		"""
        for iter_number in range(self.n_epochs):  # iterating for the specified epochs
            if self.verbose:
                print(f"Iteration {iter_number + 1}/{self.n_epochs}")
            batched_patterns = [_ for _ in zip(inputs, expected_outputs)]  # group patterns in batches
            for (batch_number, batch) in enumerate(
                    utils.chunks(batched_patterns, self.batch_size)):  # iterate over batches
                sum_of_deltas = []  # accumulator of deltas belonging to the batch
                for (pattern, expected_output) in batch:  # iterate over pattern of a single batch
                    deltas = self.__fit_pattern(pattern, expected_output)

                    # accumulate deltas of the same batch
                    if sum_of_deltas == []:
                        sum_of_deltas = deltas
                    else:
                        for index in range(len(sum_of_deltas)):
                            sum_of_deltas[index] += deltas[index]

                # at the end of batch update weights
                self.__update_weights(sum_of_deltas)  # changed from delta to sum_of_delta

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

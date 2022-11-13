from neural_network.classes.LossFunctions import LossFunction
from neural_network import utils
import numpy as np

__all__ = ["MLClassifier"]


class MLClassifier:
    def __init__(
            self,
            layers: list,
            loss: LossFunction,
            optimizer,
            batch_size: int = 100,
            learning_rate: float = 0.1,
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
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.verbose = verbose

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

        for layer in reversed_layer:
            hidden_layer_deltas = layer.backpropagate()
            deltas.insert(0, hidden_layer_deltas)

        return deltas

    def fit(self, inputs: np.array, expected_outputs: np.array) -> tuple:
        """
		:param inputs:
		:param expected_outputs:
		:return:
		"""
        train_loss = []
        train_accuracy = []
        for iter_number in range(self.n_epochs):  # iterating for the specified epochs

            train_loss.append(self.loss.f(expected_outputs.reshape(1, -1), self.predict(inputs.T)))
            train_accuracy.append(self.evaluate(inputs.T, expected_outputs.reshape(1, -1)))  # predict all the inputs together

            if self.verbose:
                print(f"Iteration {iter_number + 1}/{self.n_epochs}\tLoss {train_loss[-1]}\tAccuracy {train_accuracy[-1]}")
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
                self.optimizer.apply(self.layers, sum_of_deltas)  # changed from delta to sum_of_deltas
        return train_loss, train_accuracy

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

from neural_network.classes.LossFunctions import LossFunction
from neural_network import utils
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
        # TODO implements metric defined by users

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

    def fit(self, inputs: np.array, expected_outputs: np.array, validation_data: list = None) -> Result:
        """
		:param inputs:
		:param expected_outputs:
		:param validation_data:
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
        for iter_number in range(self.n_epochs):  # iterating for the specified epochs

            train_loss.append(np.mean(self.loss.f(expected_outputs, self.predict(inputs))))
            train_accuracy.append(self.evaluate(inputs, expected_outputs))  # predict all the inputs together

            if validation_data:
                validation_loss.append(np.mean(self.loss.f(validation_data[1], self.predict(validation_data[0]))))
                validation_accuracy.append(self.evaluate(validation_data[0], validation_data[1]))

            if self.verbose:
                print(
                    f"Iteration {iter_number + 1}/{self.n_epochs}\tLoss {train_loss[-1]:.5f}\tAccuracy {train_accuracy[-1]:.5f}",
                    end="")
                if validation_data:
                    print(f"\tval loss {validation_loss[-1]:.5f}\tval accuracy {validation_accuracy[-1]:.5f}", end="")
                print("")
            # group patterns in batches
            for (batch_number, (batch_in, batch_out)) in enumerate(
                    utils.chunks(inputs, expected_outputs, self.batch_size)):  # iterate over batches

                # deltas = self.__fit_pattern(batch_in, batch_out)
                # deltas = list(map(lambda x: np.divide(x, len(batch_out)), deltas))

                # at the end of batch update weights
                self.optimizer.apply(self, batch_in, batch_out)  # changed from delta to sum_of_deltas
        metrics = {"train_loss": train_loss[-1], "train_acc": train_accuracy[-1]}
        history = {"train_loss_curve": train_loss, "train_acc_curve": train_accuracy}
        if validation_data:
            metrics["val_loss"] = validation_loss[-1]
            metrics["val_acc"] = validation_accuracy[-1]
            history["val_loss_curve"] = validation_loss,
            history["val_acc_curve"] = validation_accuracy
        result = Result(metrics=metrics,
                        history=history)
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

    def evaluate_result(self, input: np.array, expected_output: np.array):
        output = self.predict(input)
        result = Result(metrics={"accuracy": np.mean(expected_output == np.rint(output))}, history={})
        return result
    def dump_model(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as file:
            return pickle.load(file)

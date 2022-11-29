import numpy as np

from neural_network.classes.ActivationFunctions import ActivationFunction
from neural_network.classes.Initializer import Initializer
from neural_network.classes.Layer import Layer


class InputLayer(Layer):

    def __init__(
            self,
            input_shape: tuple,
            number_units: int,
            activation_function: ActivationFunction,
            initializer: Initializer
    ) -> None:
        super().__init__(number_units, activation_function, initializer)
        self.input_shape = input_shape
        self.inputs = None

	def build(self) -> None:
		self.weights = self.initializer(shape=(self.input_shape[-1] + 1, self.number_units))
		self.built = True

    def match_shape(self, v):
        # check if shape of v match the input_shape provided
        # if a None value is present in the input_shape provided any value is admitted for that axis in the shape of v
        if len(self.input_shape) != len(v.shape):
            return False
        for s1, s2 in zip(self.input_shape, v.shape):
            if s1 is None:
                continue
            if s1 != s2:
                return False
        return True

    def feedforward(self, input_vector: np.array) -> np.array:
        if not self.built:
            raise Exception("Layer not built, add it to a network")
        if not self.match_shape(input_vector):
            raise Exception("input shape not matching the shape provided")

        # save the inputs for backpropagation
        self.inputs = input_vector

        return super().feedforward(input_vector)

    def backpropagate(
            self
    ) -> np.array:
        """
        Backpropagates the error signals from the next layer, generating the error signals of the current
        layer (and updating the internal state) and computing the deltas of the current layer incoming weights
        """

		# add the bias
		inputs = np.insert(self.inputs, 0, 1, axis=-1)

		# remove the weight corresponding to bias
		next_layer_weights = self.next_layer.weights[1:, :]

		# for each next layer node do dot product between error signal and incoming weight from current unit
		self.error_signals = np.dot(self.next_layer.error_signals, next_layer_weights.T)

        derivative_applied_on_nets = self.activation_function.derivative_f(self.nets)
        self.error_signals = np.multiply(self.error_signals, derivative_applied_on_nets)

		return np.dot(inputs.T, self.error_signals)

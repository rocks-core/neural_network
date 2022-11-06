import numpy as np
from neural_network.classes.Functions.ActivationFunctions import ActivationFunction


class Layer:
    def __init__(
            self,
            number_units: int,
            activation_function: ActivationFunction,
    ) -> None:
        self.number_units = number_units
        self.activation_function = activation_function
        # internal variables needed for backprop
        self.outputs = None  # array of values
        self.nets = None  # array of values
        self.error_signals = None  # array of values

    def feedforward(self, input_vector: np.array, layer_weights: np.array) -> np.array:
        """
        Given an input vector, performs the feedforwrd across the layer, automatically updating the
        layer nets and outputs field (a.k.a. the layer internal state)

        :param input_vector: np array of numbers with shape (1, n)
        :param layer_weights: matrices of weights,
        :return: the array containing the outputs of the units of this layer
        """
        layer_weights = layer_weights.T
        # layer_weights[i,j] is the weight that j-th neuron assign to the i-th input (the opposite of what seen at lesson)

        input_vector = np.insert(input_vector, 0, 1, axis=1)  # adding bias term to input

        # compute input times matrix weights
        self.nets = np.dot(input_vector, layer_weights)

        # for each net result apply activation function
        # can be done using np function instead of math function
        self.outputs = np.array(list(map(self.activation_function.f, self.nets)))

        return self.outputs  # note this function does not return the bias term!

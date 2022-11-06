from .ActivationFunction import ActivationFunction
import math


__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
	def __init__(self) -> None: # REMEMBER TO ADD SLOPE COEFFICIENT HERE
		sigmoid_function = lambda x: 1/(1 + pow(math.e, -x))
		super().__init__(
			sigmoid_function,
			lambda x: sigmoid_function(x)*(1-sigmoid_function(x))
		)
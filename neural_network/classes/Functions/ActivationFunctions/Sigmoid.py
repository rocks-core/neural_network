from .ActivationFunction import ActivationFunction
import numpy as np


__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
	def __init__(self):
		sigmoid = lambda v: np.ones(len(v)) / (np.ones(len(v)) + np.exp(-v))
		super().__init__(
			sigmoid,
			lambda v: sigmoid(v) * (np.ones(len(v)) - sigmoid(v))
		)

from .ActivationFunction import ActivationFunction
import numpy as np


__all__ = ["TanH"]


class TanH(ActivationFunction):
	def __init__(self) -> None:
		tanh = lambda v: (np.exp(v) - np.exp(-v)) / (np.exp(v) + np.exp(-v))
		super().__init__(
			tanh,
			lambda v: np.ones(shape=v.shape) - np.square(tanh(v))
		)
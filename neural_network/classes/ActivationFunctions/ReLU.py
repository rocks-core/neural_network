import numpy as np

from .ActivationFunction import ActivationFunction


__all__ = ["ReLU"]


class ReLU(ActivationFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda v: np.maximum(
				np.zeros(shape=v.shape),
				v
			),
			lambda v: np.where(v < 0, np.zeros(shape=v.shape), np.ones(shape=v.shape))
		)
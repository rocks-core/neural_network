import numpy as np

from .ActivationFunction import ActivationFunction


__all__ = ["ReLU"]


class ReLU(ActivationFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda v: np.maximum(
				np.zeros(len(v)),
				v
			),
			lambda v: np.where(v < 0, np.zeros(len(v)), np.ones(len(v)))
		)
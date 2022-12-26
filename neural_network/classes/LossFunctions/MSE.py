import numpy as np

from .LossFunction import LossFunction

__all__ = ["MSE"]


class MSE(LossFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda expected_outputs, real_outputs: np.mean(np.sum((expected_outputs - real_outputs) ** 2, axis=1)),
			lambda expected_outputs, real_outputs: -2 / real_outputs.shape[0] * (expected_outputs - real_outputs)
		)

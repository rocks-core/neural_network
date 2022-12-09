import numpy as np

from .LossFunction import LossFunction

__all__ = ["MSE"]


class MSE(LossFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda expected_outputs, real_outputs: 0.5 * np.average((expected_outputs - real_outputs) ** 2, axis=1),
			lambda expected_outputs, real_outputs: -(expected_outputs - real_outputs)
		)

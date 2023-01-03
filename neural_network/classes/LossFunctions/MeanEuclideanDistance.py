import numpy as np

from .LossFunction import LossFunction


class MeanEuclideanDistance(LossFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda expected_outputs, real_outputs: np.mean(np.sqrt(np.sum((expected_outputs - real_outputs) ** 2, axis=1))),
			lambda expected_outputs, real_outputs: - 1 / real_outputs.shape[0] * (expected_outputs - real_outputs) / np.linalg.norm(expected_outputs - real_outputs, axis=1).reshape(-1, 1)
		)

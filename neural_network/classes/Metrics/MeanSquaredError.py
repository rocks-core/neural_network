import numpy as np

from neural_network.classes.Metrics import Metric


class MeanSquaredError(Metric):
	def __init__(self):
		name = "mse"
		f = lambda expected_outputs, real_outputs: np.average((expected_outputs - real_outputs) ** 2, axis=1)
		super().__init__(name, f)

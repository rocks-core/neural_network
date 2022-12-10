import numpy as np

from neural_network.classes.Metrics import Metric


class MeanSquaredError(Metric):
	def __init__(self):
		name = "mse"
		f = lambda expected_outputs, outputs: np.average((expected_outputs - outputs) ** 2)
		super().__init__(name, f)

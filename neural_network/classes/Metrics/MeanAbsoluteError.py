import numpy as np

from neural_network.classes.Metrics import Metric


class MeanAbsoluteError(Metric):
	def __init__(self):
		name = "mae"
		f = lambda expected_outputs, real_outputs: np.average(np.abs(expected_outputs - real_outputs), axis=1)
		super().__init__(name, f)

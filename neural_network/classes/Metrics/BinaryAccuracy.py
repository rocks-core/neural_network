import numpy as np

from neural_network.classes.Metrics import Metric


class BinaryAccuracy(Metric):
	def __init__(self):
		name = "binary_accuracy"
		f = lambda expected_outputs, outputs: np.average((expected_outputs == np.rint(outputs)))
		super().__init__(name, f)

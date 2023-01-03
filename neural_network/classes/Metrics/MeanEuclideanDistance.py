import numpy as np

from neural_network.classes.Metrics import Metric


class MeanEuclideanDistance(Metric):
	def __init__(self):
		name = "mean_euclidean_distance"
		f = lambda expected_outputs, real_outputs: np.mean(np.linalg.norm(expected_outputs - real_outputs, axis=1))
		super().__init__(name, f)

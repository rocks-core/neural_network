from neural_network.classes.Metrics import *


class MetricConverter:
	@staticmethod
	def get_from_string(string):
		if string == "mse" or string == "mean_squared_error":
			return MeanSquaredError()
		if string == "mae" or string == "mean_absolute_error":
			return MeanAbsoluteError()
		if string == "mean_euclidean_distance":
			return MeanEuclideanDistance()
		if string == "binary_accuracy":
			return BinaryAccuracy()
		raise Exception("name of the metric not recognized")
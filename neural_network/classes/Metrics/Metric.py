from neural_network.classes.Metrics import *


class Metric:
	def __init__(self, name, f):
		self.name = name
		self.f = f

	@staticmethod
	def get_from_string(string):
		if string == "mse" or string == "mean_squared_error":
			return MeanSquaredError()
		if string == "mae" or "mean_absolute_error":
			return MeanAbsoluteError()
		if string == "mean_euclidean_distance":
			return MeanEuclideanDistance()
		raise Exception("name of the metric not recognized")

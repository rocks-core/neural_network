from neural_network.classes.Metrics import *


class Metric:
	def __init__(self, name, f):
		self.name = name
		self.f = f

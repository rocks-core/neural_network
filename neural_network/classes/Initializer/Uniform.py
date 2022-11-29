import numpy as np

from .Initializer import Initializer


class Uniform(Initializer):
	def __init__(self, low, high):
		super().__init__()
		self.low = low
		self.high = high

	def __call__(self, shape, *args, **kwargs):
		return np.random.uniform(self.low, self.high, shape)

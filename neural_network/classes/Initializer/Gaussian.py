import numpy as np

from .Initializer import Initializer


class Gaussian(Initializer):
	def __init__(self, mean, std):
		super().__init__()
		self.mean = mean
		self.std = std

	def __call__(self, shape, *args, **kwargs):
		return np.random.normal(self.mean, self.std, shape)

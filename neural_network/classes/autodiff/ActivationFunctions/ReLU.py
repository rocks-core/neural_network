import numpy as np
from .ActivationFunction import ActivationFunction


class ReLU(ActivationFunction):
	def f(self, af, x):
		return af.max(x, np.array(0.))

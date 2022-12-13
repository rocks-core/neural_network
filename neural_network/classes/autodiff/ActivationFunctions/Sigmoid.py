import math
import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
	def f(self, af, x):
		t = af.division(np.array(1.), af.sum(af.exp(math.e, af.product(x, np.array(-1.))), np.array(1.)))
		return t

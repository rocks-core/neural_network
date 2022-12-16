import math
import numpy as np
from .ActivationFunction import ActivationFunction


class Sigmoid(ActivationFunction):
	def f(self, af, x):
		return af.division(np.array(1.), af.add(af.exp(math.e, af.product(x, np.array(-1.))), np.array(1.)))

import math
import numpy as np
from .ActivationFunction import ActivationFunction


class TanH(ActivationFunction):
	def f(self, af, x):
		e_xp = af.exp(math.e, x)
		e_xm = af.exp(math.e, af.product(x, np.array(-1.)))
		return af.division(af.sub(e_xp, e_xm), af.sum(e_xp, e_xm))

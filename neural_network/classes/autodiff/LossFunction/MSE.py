from .LossFunction import LossFunction
import numpy as np


class MSE(LossFunction):
	def f(self, af, expected_output, output):
		diff = af.sub(expected_output, output)
		return af.product(np.array(0.5), af.matmul(af.transpose(diff), diff))

from neural_network.classes.autodiff.ActivationFunctions import ActivationFunction
import math


class Softmax(ActivationFunction):
	def f(self, af, x):
		return af.division(af.exp(math.e, x), af.sum(af.exp(math.e, x), ax=1))

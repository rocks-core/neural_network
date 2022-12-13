from abc import abstractmethod


class ActivationFunction:
	@abstractmethod
	def f(self, af, x):
		pass

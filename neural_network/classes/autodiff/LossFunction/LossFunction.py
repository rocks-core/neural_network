from abc import abstractmethod


class LossFunction:
	@abstractmethod
	def f(self, af, expected_output, output):
		pass

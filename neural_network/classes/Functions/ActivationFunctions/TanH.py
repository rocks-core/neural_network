from .ActivationFunction import ActivationFunction
import math


__all__ = ["TanH"]


class TanH(ActivationFunction):
	def __init__(self) -> None:
		tanh = lambda x: (math.pow(math.e, x) - math.pow(math.e, -x) / math.pow(math.e, x) + math.pow(math.e, -x))
		super().__init__(
			tanh,
			lambda x: 1 - math.pow(tanh(x), 2)
		)
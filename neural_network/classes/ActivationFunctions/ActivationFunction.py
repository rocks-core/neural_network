__all__ = ["ActivationFunction"]


class ActivationFunction:
	def __init__(
			self,
			f,
			derivative_f
	) -> None:
		self.f = f
		self.derivative_f = derivative_f

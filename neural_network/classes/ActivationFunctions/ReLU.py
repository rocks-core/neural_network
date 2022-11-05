from .ActivationFunction import ActivationFunction


__all__ = ["ReLU"]


class ReLU(ActivationFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda x: max(0, x),
			lambda x: None if x == 0 else 1 if x > 0 else 0
		)
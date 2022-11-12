from .LossFunction import LossFunction


__all__ = ["MSE"]


class MSE(LossFunction):
	def __init__(self) -> None:
		super().__init__(
			lambda expected_outputs, real_outputs: 0.5 * (expected_outputs - real_outputs) ** 2,
			lambda expected_outputs, real_outputs: (expected_outputs - real_outputs)
		)

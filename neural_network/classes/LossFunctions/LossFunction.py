__all__ = ["LossFunction"]


class LossFunction:
	def __init__(
			self,
			f,
			derivative_f
	) -> None:
		"""
		f: function, loss function; takes as input 2 np.arrays (the expected outputs and the real outputs)
			and returns a np.array where the i-th element is the application of the loss function on the
			i-th elements of the inputs
		derivative_f: function, derivative of the loss function
		"""
		self.f = f
		self.derivative_f = derivative_f

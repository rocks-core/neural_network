import itertools as it
from collections import deque


class EarlyStopping:
	def __init__(
			self,
			monitor: str,
			patience: int,
			mode: str = "min",
			min_delta: float = 0,
			baseline: float = None,
			restore_best_weight: bool = False
	):
		"""
		:param monitor: str, quantity to be monitored
		:param patience: int, number of epochs with no improvement after which training will be stopped
		:param mode: str, one of {"min", "max"}. In min mode, training will stop when the quantity monitored
		has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing
		:param min_delta: float, minimum change in the monitored quantity to qualify as an improvement
		:param baseline: float, baseline value for the monitored quantity. Training will stop if the model doesn't show
		improvement over the baseline
		:param restore_best_weight: bool, whether to restore model weights from the epoch with the best value of the
		monitored quantity
		"""
		self.monitor = monitor
		self.patience = patience
		self.mode = mode
		self.min_delta = min_delta
		self.baseline = baseline
		self.restore_best_weight = restore_best_weight
		self.monitored_values = deque([], maxlen=patience+1)

	def add_monitored_value(
			self,
			monitored_value: tuple
	) -> bool:
		"""
		Checks whatever the variations of the monitored value justify an early stopping

		:param monitored_value: tuple, the last record of the monitored value
		:return: True if the stopping conditions are justified, False otherwise
		"""
		layers, value = monitored_value

		if self.mode == "min":
			compare_function = lambda n, m: n > m
		elif self.mode == "max":
			compare_function = lambda n, m: n < m

		# immediately check the baseline value
		if self.baseline is not None and compare_function(value, self.baseline) == True:
			return True
		else:
			self.monitored_values.append(monitored_value)

			if len(self.monitored_values) < self.patience + 1:
				return False
			else:
				values_to_check = [ value for model, value in self.monitored_values ]
				# compute deltas
				deltas = [ after_value - before_value for before_value,after_value in it.pairwise(values_to_check) ]

				# check if all deltas have shown no improvements
				improvement_function = lambda n: compare_function(n, self.min_delta)
				return all(map(improvement_function, deltas))

	def get_best_weights(self) -> list:
		# order states by monitored value
		list_values = list(self.monitored_values)
		list_values.sort(key=lambda elem: elem[1])

		# return best layers configuration
		layers_index = 0 if self.mode == "min" else -1
		return list_values[layers_index][0]
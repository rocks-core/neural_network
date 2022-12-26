import itertools as it
from collections import deque


class EarlyStopping:
	def __init__(
			self,
			monitor: str,
			patience: int,
			mode: str = "min",
			min_delta: float = 0,
			restore_best_weight: bool = False
	):
		"""
		:param monitor: str, quantity to be monitored
		:param patience: int, number of epochs with no improvement after which training will be stopped
		:param mode: str, one of {"min", "max"}. In min mode, training will stop when the quantity monitored
		has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing
		:param min_delta: float, minimum change in the monitored quantity to qualify as an improvement
		:param restore_best_weight: bool, whether to restore model weights from the epoch with the best value of the
		monitored quantity
		"""
		self.monitor = monitor
		self.patience = patience
		self.mode = mode
		self.min_delta = min_delta
		self.restore_best_weight = restore_best_weight
		self.best_weights = None
		self.best_value = None
		self.call_since_best = 0

	def __call__(self, model, *args, **kwargs):
		"""
		Checks whatever the variations of the monitored metric justify an early stopping and set to True
		the early_stop attribute of the calling model if so

		:param model: model that called the callback
		"""
		if self.monitor not in model.metrics:
			return
		# check if we are seeking to minimize of maximize the metric
		if self.mode == "min":
			compare_function = lambda n, m: n + self.min_delta <= m
		elif self.mode == "max":
			compare_function = lambda n, m: n + self.min_delta >= m

		if not self.best_value or compare_function(model.metrics_history[self.monitor][-1], self.best_value):
			self.best_value = model.metrics_history[self.monitor][-1]
			self.best_weights = model.get_weights()
			self.call_since_best = 0
		else:
			self.call_since_best += 1

		if self.call_since_best > self.patience:
			if self.restore_best_weight:
				model.set_weights(self.best_weights)
			model.early_stop = True

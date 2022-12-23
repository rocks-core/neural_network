import numpy as np
class NesterovSGD:
	def __init__(self, learning_rate: float, momentum: float, regularization: float):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.regularization = regularization
		self.old_deltas = []

	def apply(self, model, x, y):
		if not self.old_deltas:
			deltas = [d / x.shape[0] for d in model.fit_pattern(x, y)]
			for layer, delta in zip(model.layers, deltas):
				layer.weights = layer.weights + self.learning_rate * delta - 2 * self.regularization * layer.weights
			self.old_deltas = deltas

		else:
			new_deltas = []

			for layer, old_delta in zip(model.layers, self.old_deltas):
				layer.weights = layer.weights + self.momentum * old_delta
			deltas = [d / x.shape[0] for d in model.fit_pattern(x, y)]
			for layer, old_delta in zip(model.layers, self.old_deltas):
				layer.weights = layer.weights - self.momentum * old_delta

			for layer, delta, old_delta in zip(model.layers, deltas, self.old_deltas):
				new_delta = self.learning_rate * delta + self.momentum * old_delta
				new_deltas.append(new_delta)
				layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2
			self.old_deltas = new_deltas

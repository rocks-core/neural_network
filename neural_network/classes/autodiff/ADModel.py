from neural_network.classes.autodiff import AutodiffFramework
from neural_network.classes.autodiff.LossFunction import LossFunction
from neural_network.classes.Model import Model


class ADModel(Model):
	def __init__(self,
	             layers: list,
	             loss: LossFunction,
	             optimizer,
	             n_epochs=100,
	             batch_size=100,
	             callbacks: list = [],
	             metrics: list = [],
	             shuffle: bool = False,
	             verbose: bool = False):
		self.af = AutodiffFramework(strict=True)
		super().__init__(layers, loss, optimizer, n_epochs, batch_size, callbacks, metrics, shuffle, verbose)

	def build_layer(self, layers, loss):
		layers[0].build(self.af)
		self.layers.append(layers[0])
		for layer in layers[1:]:
			layer.build(self.af, self.layers[-1])
			self.layers.append(layer)

	def fit_pattern(self, pattern, expected_output):
		inputs = pattern
		for layer in self.layers:
			inputs = layer(inputs)
		output = inputs
		loss = self.loss.f(self.af, expected_output, output)
		deltas = []
		for layer in reversed(self.layers):
			delta = -self.af.gradient(loss, layer.weights)
			deltas.insert(0, delta)
		self.af.reset()
		return deltas

	def predict(self, input):
		return super().predict(input).compute()

	def evaluate(self, input, expected_output):
		return super().evaluate(input, expected_output).compute()

	def get_weights(self):
		weights = []
		for layer in self.layers:
			weights.append(layer.weights.get_value())
		return weights

	def set_weights(self, weights):
		for layer, w in zip(self.layers, weights):
			layer.weights.assign(w)

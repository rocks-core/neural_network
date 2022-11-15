class SGD:
    def __init__(self, learning_rate: float, momentum: float, regularization: float, nesterov: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.nesterov = nesterov
        self.old_deltas = []

    def apply(self, layers, deltas):
        if not self.old_deltas:
            for layer, delta in zip(layers, deltas):
                layer.weights = layer.weights + self.learning_rate * delta - 2 * self.regularization * layer.weights
            self.old_deltas = list(deltas)
        else:
            if self.nesterov:
                for layer, delta, old_delta in zip(layers, deltas, self.old_deltas):
                    layer.weights = layer.weights + self.learning_rate * delta - 2 * self.regularization * layer.weights + self.momentum * old_delta
                self.old_deltas = list(deltas)
            else:
                new_deltas = []
                for layer, delta, old_delta in zip(layers, deltas, self.old_deltas):
                    new_delta = self.learning_rate * delta + self.momentum * old_delta
                    new_deltas.append(new_delta)
                    layer.weights = layer.weights + new_delta - 2 * self.regularization * layer.weights
                self.old_deltas = new_deltas

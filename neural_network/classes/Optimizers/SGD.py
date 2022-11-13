class SGD:
    def __init__(self, learning_rate: float, momentum: float, regularization: float, nesterov: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.nesterov = nesterov
        self.old_gradients = []

    def apply(self, layers, gradients):
        if not self.old_gradients:
            for layer, gradient in zip(layers, gradients):
                layer.weights = layer.weights + self.learning_rate * gradient - 2 * self.regularization * layer.weights
            self.old_gradients = list(gradients)
        else:
            for layer, gradient, old_gradient in zip(layers, gradients, self.old_gradients):
                layer.weights = layer.weights + self.learning_rate * gradient - 2 * self.regularization * layer.weights + self.momentum * old_gradient
            self.old_gradients = list(gradients)
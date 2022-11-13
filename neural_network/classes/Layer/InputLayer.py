from neural_network.classes.Layer import Layer

class InputLayer:

    def __init__(self, input_size):
        self.input_size = input_size
        self.number_units = input_size
        self.outputs = None
        self.next_layer = None

    def feedforward(self, inputs):
        self.outputs = inputs
        return self.outputs
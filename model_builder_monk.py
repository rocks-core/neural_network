from neural_network.classes.Model import Model
from neural_network.classes.Layers import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Initializer import Uniform
from neural_network.classes.ActivationFunctions import Sigmoid

def model_builder(config : dict):
    """
    Given a cofiguration, i.e. a dict that maps the hyperparameter name with the current value, this function built the model
    
    :param verbose: verbose, it is also passed to the model
    :input shape: the size of the input vector (input layer)
    :output shape: the size of the output vector (output layer) 
    """
    verbose = False
    input_shape = 17
    output_shape = 1

    num_layer = config["num_hidden_layers"]

    # check if the layers before num_hidden_layers are correct
    for i in range(1, num_layer+1):
        if config["neurons_in_layer_"+str(i)] == 0:
            if verbose:
                print(f"model build failed because the model has {num_layer} layers but the layer {i} has 0 neurons")
            return None

    # check if all the layers after num_hidden_layers have zero units
    i = 1
    while "neurons_in_layer_"+str(num_layer + i) in config.keys():
        if config["neurons_in_layer_"+str(num_layer + i)] != 0:
            if verbose:
                print(f"model build failed because the model has {num_layer} layers but layer {num_layer + i} has not 0 neurons")
            return None
        i +=1

    # built the layers
    layers = []
    layers.append(InputLayer((None, input_shape), config["neurons_in_layer_1"], config["activation_function"](), initializer=Uniform(-1, 1)))

    for i in range(2, num_layer+1):
        layers.append(HiddenLayer( config["neurons_in_layer_"+str(i)], config["activation_function"](), initializer=Uniform(-1, 1)))

    layers.append(OutputLayer(output_shape, Sigmoid(), initializer=Uniform(-1, 1)))

    model = Model(
			layers = layers,
			loss = config["loss_function"](),
			optimizer = config["optimizer"](
                config["learning_rate"],
                config["momentum"],
                config["regularization"]
            ),
            metrics=["mse", "binary_accuracy"],
			batch_size=config["batch_size"],
			n_epochs=config["num_epochs"],
            callbacks=config["callbacks"],
			verbose=verbose
		)

    if verbose:
        print(f"model builded")

    return model
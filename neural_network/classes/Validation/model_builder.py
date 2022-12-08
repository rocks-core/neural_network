
from neural_network.classes import ActivationFunctions
from neural_network.classes.MLClassifier import MLClassifier
from neural_network.classes.Layer import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Initializer import Uniform

def model_builder(config : dict, verbose : bool, input_shape : int, output_shape : int):
    """
    Given a cofiguration, i.e. a dict that maps the hyperparameter name with the current value, this function built the model
    
    :param verbose: verbose, it is also passed to the model
    :input shape: the size of the input vector (input layer)
    :output shape: the size of the output vector (output layer) 
    """
    
    num_layer = config["num_hidden_layers"]

    #check if the layers before num_hidden_layers are correct
    for i in range(1, num_layer+1):
        if config["neurons_in_layer_"+str(i)] == 0:
            if verbose:
                print(f"model build failed because the model has {num_layer} layers but the layer {i} has 0 neurons")
            return None

    #check if all the layers after num_hidden_layers have zero units
    i = 1
    while "neurons_in_layer_"+str(num_layer + i) in config.keys():
        if config[ "neurons_in_layer_"+str(num_layer + i) ] != 0:
            if verbose:
                print(f"model build failed because the model has {num_layer} layers but layer {num_layer + i} has not 0 neurons")
            return None
        i +=1
    
    #built the layers
    layers = []
    layers.append( InputLayer((None, input_shape), input_shape, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)) )

    for i in range(1, num_layer+1):
        layers.append( HiddenLayer( config["neurons_in_layer_"+str(i)], ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)))

    layers.append(OutputLayer(output_shape, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)))

    model = MLClassifier(
			layers = layers,
			loss = config["loss_function"](),
			optimizer = config["optimizer"](
                config["optimizer_learning_rate"], 
                config["optimizer_momentum"], 
                config["optimizer_regularization"]
            ),
			batch_size=config["batch_size"],
			n_epochs=config["num_epochs"],
			verbose=verbose
		)
    
    if verbose:
        print(f"model builded")
    
    return model
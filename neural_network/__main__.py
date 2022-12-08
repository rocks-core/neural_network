from neural_network import LossFunctions
from neural_network import datasets
from neural_network.classes.Optimizers import NesterovSGD
from neural_network.classes.Validation import ConfigurationGenerator, Hyperparameter, model_builder
import neural_network.utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

def main():
	


	
	

	





generator = ConfigurationGenerator(
	mode = "grid",
	num_trials = 1,
	folded_hyper_space={
		
		"loss_function" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ LossFunctions.MSE ]
		),
		"optimizer" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ NesterovSGD ]
		),
		"optimizer_learning_rate" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 0.1 ]
		),
		"optimizer_momentum" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 0.9 ]
		),
		"optimizer_regularization" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 0. ]
		),
		"batch_size" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 200 ]
		),
		"num_epochs" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 1000 ]
		),
		"num_hidden_layers" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 3 ]
		),
		"neurons_in_layer_1" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 30 ]
		),
		"neurons_in_layer_2" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 30 ]
		),
		"neurons_in_layer_3" : Hyperparameter(
				generator_logic="all_from_list",
				generator_space=[ 30 ]
		)
	}
)



if __name__ == "__main__":
	
	main()
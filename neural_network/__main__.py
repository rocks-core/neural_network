from neural_network import ActivationFunctions
from neural_network import LossFunctions
from neural_network import MLClassifier
from neural_network import datasets
from neural_network.classes.Layer import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Optimizers import SGD
from neural_network import Validation
from neural_network.classes.Initializer import Uniform
from neural_network.classes.Validation import Hyperparameter, ConfigurationGenerator
import neural_network.utils
import numpy as np

## test
cg = ConfigurationGenerator(
    {
        "num_layers" : Hyperparameter(
            generator_logic = "all_from_list", 
            generator_space = [4, 5, 6, 7]),
        
        "lambda" : Hyperparameter(
            generator_logic = "random_choice_from_range", 
            generator_space = (0.1, 0.6), 
            random_elements_to_generate = 2),
    })

    
for config in cg:
	print(config)


if __name__ == "__main__":
	dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
	dataset_class_column = "class"
	number_inputs = len(dataset_attribute_columns)
	loss_function = LossFunctions.MSE()

	"""
	tr_df, vl_df, _ = neural_network.utils.split_samples(
		df=datasets.read_monk1()[0],
		tr_size=0.7,
		vl_size=0.3,
		ts_size=0.0
	)
	n_trials = 5

	tr_inputs = tr_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
	tr_outputs = tr_df[dataset_class_column].to_numpy()
	vl_inputs = vl_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
	vl_outputs = vl_df[dataset_class_column].to_numpy()
	"""
	tr_df, _, _ = neural_network.utils.split_samples(
		df=datasets.read_monk1()[0],
		tr_size=1,
		vl_size=0.0,
		ts_size=0.0
	)
	inputs = tr_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
	outputs = tr_df[dataset_class_column].to_numpy()

	layers = [
		InputLayer((None, inputs.shape[-1]), 5, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
		HiddenLayer(8, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
		OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1))
	]

	number_folds = 4
	k_fold = Validation.K_fold(
		len(inputs),
		number_folds
	)

	folds_score = []
	for (fold_tr_indexes, fold_vl_indexes) in k_fold.get_folds():
		fold_tr_inputs, fold_tr_outputs = inputs[fold_tr_indexes], outputs[fold_tr_indexes]
		fold_vl_inputs, fold_vl_outputs = inputs[fold_vl_indexes], outputs[fold_vl_indexes]

		classifier = MLClassifier(
			layers=layers,
			loss=loss_function,
			optimizer=SGD(learning_rate=0.05, momentum=0., regularization=0.0),
			batch_size=100,
			n_epochs=1000,
			verbose=False
		)
		# training model
		classifier.fit(
			fold_tr_inputs,
			fold_tr_outputs,
			validation_data=(fold_vl_inputs, fold_vl_outputs)
		)
		print("Done training")

		# validating result
		correct_predictions = 0

		folds_score.append(100 * classifier.evaluate(fold_vl_inputs, fold_vl_outputs))

	print(f"fold scores: {folds_score}")
	print(f"min: {min(folds_score)}")
	print(f"max: {max(folds_score)}")
	avg = lambda l: sum(l) / len(l) if len(l) != 0 else 0
	print(f"avg: {avg(folds_score)}")

	"""
	trials = []
	for _ in range(n_trials):
		classifier = MLClassifier(
			layers=layers,
			loss=loss_function,
			optimizer=SGD(learning_rate=0.05, momentum=0., regularization=0.),
			batch_size=100,
			n_epochs=1000,
			verbose=True
		)
		# training model
		classifier.fit(tr_inputs, tr_outputs, validation_data=(vl_inputs, vl_outputs))
		print("Done training")

		# validating result
		correct_predictions = 0

		trials.append(100 * classifier.evaluate(vl_inputs, vl_outputs))

	print(trials)
	print(f"min: {min(trials)}")
	print(f"max: {max(trials)}")
	avg = lambda l: sum(l) / len(l) if len(l) != 0 else 0
	print(f"avg: {avg(trials)}")
	"""

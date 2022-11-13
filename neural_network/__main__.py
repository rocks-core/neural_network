from neural_network import ActivationFunctions
from neural_network import LossFunctions
from neural_network import MLClassifier
from neural_network import datasets
from neural_network.classes.Layer import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Optimizers import SGD
from neural_network.classes.Initializer import Uniform
import neural_network.utils
import numpy as np


if __name__ == "__main__":
	dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
	dataset_class_column = "class"
	number_inputs = len(dataset_attribute_columns)
	loss_function = LossFunctions.MSE()

	tr_df, vl_df, _ = neural_network.utils.split_samples(
		df=datasets.read_monk1()[0],
		tr_size=0.7,
		vl_size=0.3,
		ts_size=0.0
	)
	n_trials = 5

	tr_inputs = tr_df[dataset_attribute_columns].to_numpy()
	tr_outputs = tr_df[dataset_class_column].to_numpy()
	vl_inputs = vl_df[dataset_attribute_columns].to_numpy()
	vl_outputs = vl_df[dataset_class_column].to_numpy()

	layers = [
			InputLayer((tr_inputs.shape[-1], 1), 2, ActivationFunctions.Linear(), initializer=Uniform(-0.1, 0.1)),
			HiddenLayer(2, ActivationFunctions.Linear(), initializer=Uniform(-0.1, 0.1)),
			OutputLayer(1, ActivationFunctions.Sigmoid(), loss_function, initializer=Uniform(-0.1, 0.1))
	]

	trials = []
	for _ in range(n_trials):
		classifier = MLClassifier(
			layers=layers,
			optimizer=SGD(learning_rate=0.01, momentum=0.001, regularization=0.001),
			batch_size=100,
			learning_rate=0.001,
			n_epochs=200,
			verbose=False
		)
		# training model
		classifier.fit(tr_inputs, tr_outputs)
		print("Done training")

		# validating result
		correct_predictions = 0

		for (input, expected_output) in zip(vl_inputs, vl_outputs):
			input = input.reshape(-1, 1)  # inputs have to be row vector shape (n, 1)
			real_output = classifier.predict(input)[0, 0]  # output is a (1,1) matrix now

			if round(real_output) == expected_output:
				correct_predictions += 1

		trials.append(100 * (correct_predictions / len(vl_df)))

	print(trials)
	print(f"min: {min(trials)}")
	print(f"max: {max(trials)}")
	avg = lambda l: sum(l)/len(l) if len(l) != 0 else 0
	print(f"avg: {avg(trials)}")


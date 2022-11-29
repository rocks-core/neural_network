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

	tr_inputs = tr_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
	tr_outputs = tr_df[dataset_class_column].to_numpy()
	vl_inputs = vl_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
	vl_outputs = vl_df[dataset_class_column].to_numpy()

	layers = [
		InputLayer((None, tr_inputs.shape[-1]), 5, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
		HiddenLayer(8, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
		OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1))
	]

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

	dict = {"unit_1": hp.Choice([4, 10, 16]), "learning_rate": hp.Float([0.0001, 0.1], 5), "unit_2": hp.Int([30, 50])}
	for hp in tuner(dict):
		yield model_builder(hp)

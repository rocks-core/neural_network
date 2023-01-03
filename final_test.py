from model_builder import model_builder
from neural_network.classes.Callbacks import EarlyStopping
from neural_network.classes.LossFunctions import MeanEuclideanDistance
from neural_network.classes.Optimizers import *
from neural_network.classes.Validation import *
import pandas as pd
from neural_network.classes.ActivationFunctions import Sigmoid, TanH
from neural_network.classes.Validation import TunerCV


dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]
dataset_class_column = ["target_x", "target_y"]
dataset = pd.read_csv("neural_network/datasets/MLCup/train.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

dataset_y = dataset[dataset_class_column].to_numpy()
dataset_x = dataset[dataset_attribute_columns].to_numpy()

dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]
dataset_class_column = ["target_x", "target_y"]
dataset = pd.read_csv("neural_network/datasets/MLCup/test.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

test_set_y = dataset[dataset_class_column].to_numpy()
test_set_x = dataset[dataset_attribute_columns].to_numpy()

result = ResultCollection.load("dumps/model_assessment_validation/fine_search.pickle")
result.sort("val_mean_euclidean_distance", False)
for i in range(1, 5, 1):
	best_hp = result.list[i].hp_config
	best_model = model_builder(best_hp)
	best_model.verbose = True
	refit_results = best_model.fit(dataset_x, dataset_y, epochs=1000, batch_size=200)
	test_result = best_model.evaluate_result(test_set_x, test_set_y)

	refit_results.dump(f"./dumps/model_assessment_validation/refit_results_best_{i}.pickle")
	test_result.dump(f"./dumps/model_assessment_validation/test_results_best_{i}.pickle")
	# best_model.dump_weights("./dumps/best_weights.pickle")

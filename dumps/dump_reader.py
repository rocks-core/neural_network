from neural_network.classes.Results import *


monk_result = TestResult.load("./monk_results/validation_monk1_results.pickle")
monk_result.plot_one(1, "mse", "val_mse")

# result = TestResult.load("./monk1_results.pickle")
# result = 0

# test_results = TestResult.load("model_assesment_validation/test_results.pickle")
refit_result = Result.load("final_refit_all_dataset/refit_results.pickle")
refit_result.plot("mean_euclidean_distance")


result = ResultCollection.load("model_assesment_validation/fine_search.pickle")
result.sort("val_mean_euclidean_distance", False)
for r in result.list:
	print(r.hp_config["learning_rate"])
result.plot("mean_euclidean_distance", "val_mean_euclidean_distance")

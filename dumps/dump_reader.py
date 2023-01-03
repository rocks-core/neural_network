from neural_network.classes.Results import *

# scaled_results = ResultCollection.load("./fine_search_scaled.pickle")
# scaled_results.plot("mean_euclidean_distance", "val_mean_euclidean_distance")

# for i in range(1, 4, 1):
# 	monk_result = Result.load(f"dumps/monk_results/monk{i}/test_results.pickle")
# 	monk_result.validation_results.plot_one(0, "mse", "val_mse", save_path=f"Plots/monk{i}/mse")
# 	monk_result.validation_results.plot_one(0, "binary_accuracy", "val_binary_accuracy", save_path=f"Plots/monk{i}/bin_acc")

# result = TestResult.load("./monk1_results.pickle")
# result = 0

test_results = TestResult.load("dumps/model_assessment_validation/test_results.pickle")
# test_results.validation_results.plot_one(0, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/best", show=False)
# test_results.validation_results.plot_one(1, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/best2", show=False)
# test_results.validation_results.plot_one(2, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/best3", show=False)
# test_results.validation_results.plot_one(3, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/best4", show=False)
# test_results.validation_results.plot_one(4, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/best5", show=False)
# refit = Result.load("dumps/model_assessment_validation/refit_results.pickle")
# refit.plot("mean_euclidean_distance", title="best_refit_mee.png", save_path="Plots/cup")

#
# grid_search_results = ResultCollection.load("dumps/final_refit_all_dataset/grid_search_results.pickle")
# grid_search_results.plot_one(0, "mean_euclidean_distance", "val_mean_euclidean_distance", save_path="Plots/cup/final/best_model", show=False)
# refit_result = Result.load("dumps/final_refit_all_dataset/refit_results.pickle")
# refit_result.plot("mean_euclidean_distance", title="refit", save_path="Plots/cup/final", show=False)


coarse_result = ResultCollection.load("dumps/model_assessment_validation/coarse_search.pickle")
coarse_result.sort("val_mean_euclidean_distance", False)

for i, r in enumerate(coarse_result.list):
	if r.hp_config["regularization"] == 1e-2:
		print(i)

while True:
	n = int(input())
	# coarse_result.plot_one(n, "mean_euclidean_distance", "val_mean_euclidean_distance")
	print(coarse_result.list[n].hp_config)
# for i in range(len(coarse_result.list)):
# 	coarse_result.plot_one(i, "mean_euclidean_distance", "val_mean_euclidean_distance", title=f"trial_{i}.png", save_path="Plots/cup/coarse", show=False)

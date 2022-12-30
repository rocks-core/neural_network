from neural_network.classes.Results import *

# result = TestResult.load("./monk1_results.pickle")
# result = 0

result = ResultCollection.load("./coarse_search.pickle")
result.sort("val_mean_euclidean_distance", False)
for r in result.list:
	print(r.hp_config["num_hidden_layers"])
result.plot("mean_euclidean_distance", "val_mean_euclidean_distance")

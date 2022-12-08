from neural_network.classes.Validation import TunerCV
from neural_network.classes.Validation import K_fold
from neural_network.classes.Results import *


class TesterCV:
	"""
	Class to perform model assessment using K-fold cross validation
	"""

	def __init__(self,
	             tuner: TunerCV,
	             n_fold: int,
	             verbose: bool
	             ):
		"""
		Args:
			tuner: the tuner to be used to do model selection with the trainval set
			n_fold (int): number of fold
            verbose (bool): for the moment do nothing
		"""
		self.tuner = tuner
		self.n_fold = n_fold
		self.results = None
		self.verbose = verbose  # TODO use verbose somewhere or remove it

	def fit(self, inputs, outputs):  # TODO  parallelize
		k_fold = K_fold(inputs.shape[0], self.n_fold)
		evaluation_results = []
		val_results = []
		for (fold_trainval_indexes, fold_test_indexes) in k_fold.get_folds():
			# select test and trainval set
			fold_trainval_inputs, fold_trainval_outputs = inputs[fold_trainval_indexes], outputs[fold_trainval_indexes]
			fold_test_inputs, fold_test_outputs = inputs[fold_test_indexes], outputs[fold_test_indexes]

			# fit the tuner with the trainval
			val_result = self.tuner.fit(fold_trainval_inputs, fold_trainval_outputs)

			# get the model with the best hyperparameters obtained in the folds and refit it
			model = self.tuner.best_model("val_acc", True)
			model.fit(fold_trainval_inputs, fold_trainval_outputs)

			# assess the model risk on the test set
			evaluation_result = model.evaluate_result(fold_test_inputs, fold_test_outputs)
			evaluation_results.append(evaluation_result)
			val_results.append(val_result)

		self.results = TestResult(evaluation_results, val_results)
		return self.results

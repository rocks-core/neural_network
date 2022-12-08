from neural_network.classes.Validation import TunerCV
from neural_network.classes.Validation import K_fold


class TesterCV:
	def __init__(self,
				 tuner: TunerCV,
				 k_fold: K_fold,
				 verbose: bool
				 ):
		self.tuner = tuner
		self.k_fold = k_fold
		self.results = []
		self.verbose = verbose

	def fit(self, inputs, outputs):  # TODO  parallelize
		fold_results = []
		for (fold_trainval_indexes, fold_test_indexes) in self.k_fold.get_folds():
			# select test and trainval set
			fold_trainval_inputs, fold_trainval_outputs = inputs[fold_trainval_indexes], outputs[fold_trainval_indexes]
			fold_test_inputs, fold_test_outputs = inputs[fold_test_indexes], outputs[fold_test_indexes]

			# fit the tuner with the trainval
			self.tuner.fit(fold_trainval_inputs, fold_trainval_outputs)

			# get the model with the best hyperparameters obtained in the folds
			model = self.tuner.best_model("val_loss", True)

			# assess the model risk on the testset
			evaluation_result = model.evaluate(fold_test_inputs, fold_test_outputs)
			self.results.append((model, evaluation_result))
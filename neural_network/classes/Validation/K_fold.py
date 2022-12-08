from neural_network import utils

class K_fold:
	def __init__(
		self,
		number_elements: int,
		n_splits: int
	):
		self.number_elements = number_elements
		self.n_splits = n_splits

	def get_folds(self) -> tuple:
		"""
		Returns an iterator	over the folds

		:return: a tuple containing the training set indexes and the validation set indexes
		"""
		for fold in utils.get_folds(self.number_elements, self.n_splits):
			yield fold

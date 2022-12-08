from neural_network import utils
import numpy as np


class Double_K_fold:
	def __init__(
		self,
		number_elements: int,
		n_external_splits: int,
		n_internal_splits: int
	):
		self.number_elements = number_elements
		self.n_external_splits = n_external_splits
		self.n_internal_splits = n_internal_splits

	def get_folds(self) -> tuple:
		"""
		Returns an iterator	over the folds

		:return: a tuple containing the training set indexes and the validation set indexes
		"""
		# external fold
		for external_fold in utils.get_folds(self.number_elements, self.n_external_splits):
			main_part, counter_part = external_fold
			unified_counter_part = np.concatenate(counter_part)

			# internal folds in the counter part
			for internal_fold in utils.get_folds(len(unified_counter_part), self.n_internal_splits):
				internal_main_indexes, internal_counter_indexes = internal_fold
				internal_main_part = unified_counter_part[internal_main_indexes]
				internal_counter_part = unified_counter_part[internal_counter_indexes]

				yield main_part, internal_main_part, internal_counter_part
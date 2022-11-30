import math


class K_fold:
	def __init__(
		self,
		number_elements: int,
		n_splits: int
	):
		self.number_elements = number_elements
		self.n_splits = n_splits

	def get_folds(self):
		"""
		Returns an iterator	over the folds

		:return: a tuple containing the training set indexes and the validation set indexes
		"""
		elements_per_fold = math.ceil(self.number_elements / self.n_splits)

		indexes = list(range(self.number_elements))
		for i in range(0, self.number_elements, elements_per_fold):
			if i == 0:
				yield indexes[elements_per_fold:], indexes[:elements_per_fold]
			elif i == self.number_elements - elements_per_fold:
				yield indexes[:i], indexes[i:]
			else:
				yield indexes[:i] + indexes[i+elements_per_fold:], indexes[i:i+elements_per_fold]

import numpy as np
import pandas as pd
import math
from collections import deque


__all__ = [
	"chunks",
	"split_samples",
	"get_folds"
]


def chunks(inputs, expected_outputs, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(inputs), n):
		yield inputs[i:i + n], expected_outputs[i:i + n]


def split_samples(
		df: pd.DataFrame,
		tr_size: float = 0.5,
		vl_size: float = 0.25,
		ts_size: float = 0.25,
		shuffle: bool = False
) -> tuple:
	"""
	Splits a dataframe in training, validation and test set

	:param df: pandas DataFrame, the dataframe to split
	:param tr_size: float, percentage of samples to put in the training set
	:param vl_size: float, percentage of samples to put in the validation set
	:param ts_size: float, percentage of samples to put in the test set
	:param shuffle: bool, True if wants to shuffle the samples before partitioning them
	:return: a tuple containing 3 pandas DataFrame: training, validation and test set
	"""
	columns = df.columns

	# randomly shuffling points
	samples = df.values

	if shuffle:
		np.random.shuffle(samples)  # beware: shuffles values in place

	# splitting points in tr, vl and ts set
	tr_index = round(len(samples) * tr_size)
	vl_index = tr_index + round(len(samples) * vl_size)
	ts_index = vl_index + round(len(samples) * ts_size)

	train_set_points = samples[0:tr_index]
	validation_set_points = samples[tr_index:vl_index]
	test_set_points = samples[vl_index:ts_index]

	train_set_df = pd.DataFrame(data=train_set_points, columns=columns)
	validation_set_df = pd.DataFrame(data=validation_set_points, columns=columns)
	test_set_df = pd.DataFrame(data=test_set_points, columns=columns)

	return train_set_df, validation_set_df, test_set_df

"""
def get_folds(number_elements: int, n_splits: int) -> tuple:
	elements_per_fold = math.ceil(number_elements / n_splits)
	indexes = list(range(number_elements))
	for i in range(0, number_elements, elements_per_fold):
		if i == 0:
			yield indexes[elements_per_fold:], indexes[:elements_per_fold]
		elif i == number_elements - elements_per_fold:
			yield indexes[:i], indexes[i:]
		else:
			yield indexes[:i] + indexes[i + elements_per_fold:], indexes[i:i + elements_per_fold]
"""

def get_folds(number_elements: int, n_splits: int) -> tuple:
	"""
	:param number_elements: int, number of elements that we want to fold
	:param n_splits: int, number of desidered splits for the data
	:return: a tuple containing the i-th fold as first element and its counterpart as second element
	"""
	indexes = range(number_elements)
	folds = deque(np.array_split(indexes, n_splits))
	# iterate over the folds
	for i in range(len(folds)):
		main_elem = folds[0] # the i-th fold
		other_elems = [folds[j] for j in range(1, len(folds))] # its counterpart
		yield main_elem, other_elems
		folds.rotate(1) # rotate the folds in order to get the next one

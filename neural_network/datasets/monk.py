import pandas as pd


__all__ = [
	"read_monk1",
	"read_monk2",
	"read_monk3"
]


def read_monk(index: int) -> tuple:
	"""
	:param index: int, number of the monks dataset to use (e.g. monks-1.train, where 1 is the index)
	:return: tuple of pandas DataFrames, the first element is the training set and the latter is the test set
	"""
	train_set_filepath = f"neural_network/datasets/monks-{index}.train"
	test_set_filepath = f"neural_network/datasets/monks-{index}.train"

	# reading csvs
	train_set_df = pd.read_csv(
		train_set_filepath,
		sep=" ",
		names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
	).set_index("id")

	test_set_df = pd.read_csv(
		test_set_filepath,
		sep=" ",
		names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
	).set_index("id")

	return train_set_df, test_set_df


def read_monk1() -> tuple:
	return read_monk(1)


def read_monk2() -> tuple:
	return read_monk(2)


def read_monk3() -> tuple:
	return read_monk(3)

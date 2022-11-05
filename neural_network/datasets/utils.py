import random
import pandas as pd


def split_samples(
		points: list,
		tr_size: float = 0.5,
		vl_size: float = 0.25,
		ts_size: float = 0.25,
		shuffle: bool = True
) -> tuple:
	# randomly shuffling points
	if shuffle:
		random.shuffle(points)

	# splitting points in tr, vl and ts set
	tr_index = round(len(points) * tr_size)
	vl_index = tr_index + round(len(points) * vl_size)
	ts_index = vl_index + round(len(points) * ts_size)

	train_set_points = points[0:tr_index]
	validation_set_points = points[tr_index:vl_index]
	test_set_points = points[vl_index:ts_index]

	train_set_df = pd.DataFrame(data=train_set_points, columns=["x", "y", "class"])
	validation_set_df = pd.DataFrame(data=validation_set_points, columns=["x", "y", "class"])
	test_set_df = pd.DataFrame(data=test_set_points, columns=["x", "y", "class"])

	return train_set_df, validation_set_df, test_set_df

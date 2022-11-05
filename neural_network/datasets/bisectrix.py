import itertools
from .utils import split_samples


__all__ = ["bisectrix"]


def bisectrix(
		tr_size: float = 0.5,
		vl_size: float = 0.25,
		ts_size: float = 0.25
) -> tuple:
	possible_coord_values = [_ for _ in range(15)]

	# generating points
	point_class = lambda term1, term2: 1 if term1 <= term2 else 0
	points = [
		[x, y, point_class(x, y)]
		for (x, y) in itertools.product(possible_coord_values, possible_coord_values)
	]

	return split_samples(points, tr_size, vl_size, ts_size, True)

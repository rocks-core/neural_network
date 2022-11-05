import itertools
from .utils import split_samples

__all__ = ["circle"]


def circle(
		tr_size: float = 0.5,
		vl_size: float = 0.25,
		ts_size: float = 0.25
	) -> tuple:
	"""
	Set with _ points with x,y <= 9
	there is a circle of equation x^2 + y^2 = 5^2; all the points with  x^2 + y^2 <= 5^2 belongs to class 1
	:return:
	"""
	possible_coord_values = [_ for _ in range(-10, 10)]
	radius = 5

	# generating points
	point_class = lambda term1, term2: 1 if term1**2 + term2**2 <= radius**2 else 0
	points = [
		[x, y, point_class(x, y)]
		for (x, y) in itertools.product(possible_coord_values, possible_coord_values)
	]

	return split_samples(points, tr_size, vl_size, ts_size, True)

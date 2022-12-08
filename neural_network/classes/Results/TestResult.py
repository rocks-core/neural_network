import pickle
import numpy as np

class TestResult:
	"""
	Class that hold the result of the assessment of a model
	"""

	def __init__(self, results, validation_results, name="", comments=""):
		"""
		Args:
		metrics: results obtained on the test set
		validation_results (ResultCollection): Collection of result obtained during model selection
		name (str, optional): optional name to recognize the results
		comments (str, optional): optional comments of the results
		"""
		self.metrics = {}
		for m in results[0].metrics.keys():
			self.metrics[m] = np.array([r.metrics[m] for r in results]).mean()
		self.validation_results = validation_results
		self.name = name
		self.comments = comments

	def dump(self, path):
		with open(path, "wb") as file:
			pickle.dump(self, file)

	@staticmethod
	def load(path):
		with open(path, "rb") as file:
			return pickle.load(file)

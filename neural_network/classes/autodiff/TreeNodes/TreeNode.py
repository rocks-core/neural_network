from abc import ABC, abstractmethod


class TreeNode(ABC):
	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def compute(self):
		pass

	@abstractmethod
	def jacobian(self, var):
		pass

	@abstractmethod
	def backward(self, var):
		pass

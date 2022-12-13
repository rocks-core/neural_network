import numpy as np

from neural_network.classes.autodiff.TreeNodes.TreeNode import TreeNode


class ConstantNode(TreeNode):
	def __init__(self, value):
		self.value = value
		self.shape = value.shape

	def compute(self):
		return self.value.copy()

	def jacobian(self, var):
		return np.zeros(shape=var.shape + self.shape)

	def backward(self, var):
		return np.zeros(shape=var.shape)

import numpy as np

from neural_network.classes.autodiff.TreeNodes import TreeNode


class VariableNode(TreeNode):
	def __init__(self, variable):
		self.variable = variable
		self.shape = variable.shape
		self.jacobian_right = None

	def compute(self):
		return self.variable.get_value()

	def jacobian(self, var):
		if self.jacobian_right is not None and var == self.variable:
			return self.jacobian_right

		identity = np.zeros(shape=var.shape + self.shape)
		if var != self.variable:
			return identity

		for i in np.ndindex(*var.value.shape):
			identity[i + i] = 1

		self.jacobian_right = identity
		return identity

	def backward(self, var):
		if var == self.variable:
			return np.ones(shape=self.shape)
		else:
			return np.zeros(shape=self.shape)

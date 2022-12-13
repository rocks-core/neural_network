import numpy as np
from abc import abstractmethod

from neural_network.classes.autodiff.TreeNodes.TreeNode import TreeNode


def jacobian_to_gradient(jacobian, var):
	axes = tuple(a for a in range(var.value.ndim, jacobian.ndim, 1))
	return np.sum(jacobian, axis=axes)


class OperationNode(TreeNode):
	@abstractmethod
	def __init__(self):
		pass


class TransposeNode(OperationNode):
	def __init__(self, x):
		self.x = x
		self.shape = tuple(reversed(x.shape))
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.transpose(self.x.compute())
		return self.computed

	def jacobian(self, var):
		jacobian = self.x.jacobian(var)
		axes = tuple(a for a in range(var.value.ndim)) + tuple(
			a for a in reversed(range(var.value.ndim, jacobian.ndim, 1)))
		return np.transpose(jacobian, axes=axes)

	def backward(self, var):
		return np.transpose(self.x.backward(var))


class SumNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = self.x.compute() + self.y.compute()
		return self.computed

	def jacobian(self, var):
		return self.x.jacobian(var) + self.y.jacobian(var)

	def backward(self, var):
		return self.x.backward(var) + self.y.backward(var)


class SubNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = self.x.compute() - self.y.compute()
		return self.computed

	def jacobian(self, var):
		return self.x.jacobian(var) - self.y.jacobian(var)

	def backward(self, var):
		return self.x.backward(var) - self.y.backward(var)


class ProductNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = self.x.compute() * self.y.compute()
		return self.computed

	def jacobian(self, var):
		return self.x.compute() * self.y.jacobian(var) + self.y.compute() * self.x.jacobian(var)

	def backward(self, var):
		jacobian = self.x.jacobian(var) * self.y.compute() + self.y.jacobian(var) * self.x.compute()
		return jacobian_to_gradient(jacobian, var)


class MatMulNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = tuple(x.shape[:-1]) + (y.shape[-1],)
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.matmul(self.x.compute(), self.y.compute())
		return self.computed

	def jacobian(self, var):
		j_x = self.x.jacobian(var)
		j_y = self.y.jacobian(var)
		c_x = self.x.compute()
		c_y = self.y.compute()
		m1 = np.matmul(j_x, c_y)
		m2 = np.matmul(c_x, j_y)
		return m1 + m2

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class GetNode(OperationNode):
	def __init__(self, x, i):
		self.x = x
		self.i = i
		self.shape = tuple(a for a, s in zip(x.shape, i) if isinstance(s, int))
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = self.x.compute()[self.i]
		return self.computed

	def jacobian(self, var):
		jacobian = self.x.jacobian(var)
		indexes = self.i
		for _ in range(var.value.ndim):
			indexes = tuple([slice(None, None, None)]) + indexes
		return jacobian[indexes]

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


# class InsertNode(OperationNode):
# 	def __init__(self, x, y, i, ax):
# 		self.x = x
# 		self.y = y
# 		self.i = i
# 		self.ax = ax
# 		self.x_computed = None
# 		self.y_computed = None
#
# 	def compute(self):
# 		self.x_computed = self.x.compute()
# 		self.y_computed = self.y.compute()
# 		return np.insert(self.x_computed, self.i, self.y_computed, self.ax)
#
# 	def jacobian(self, var):
# 		jacobian_x = self.x.jacobian(var)
# 		jacobian_y = self.y.jacobian(var)
# 		return np.insert(jacobian_x, self.i, jacobian_y, var.value.ndim + self.ax)
#
# 	def backward(self, var):
# 		jacobian = self.jacobian(var)
# 		return jacobian_to_gradient(jacobian, var)


class ConcatNode(OperationNode):
	def __init__(self, x, y, ax):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.ax = ax
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.concatenate([self.x.compute(), self.y.compute()], axis=self.ax)
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		jacobian_y = self.y.jacobian(var)
		return np.concatenate([jacobian_x, jacobian_y], axis=var.value.ndim + self.ax)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class StackNode(OperationNode):
	def __init__(self, arr, ax):
		self.arr = arr
		self.shape = arr[0].shape[:ax] + (len(arr),) + arr[0].shape[ax:]
		self.ax = ax
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.stack([x.compute() for x in self.arr], axis=self.ax)
		return self.computed

	def jacobian(self, var):
		jacobians = [x.jacobian(var) for x in self.arr]
		return np.stack(jacobians, axis=var.value.ndim + self.ax)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class ExpNode(OperationNode):
	def __init__(self, b, x):
		self.b = b
		self.x = x
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.power(self.b, self.x.compute())
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		return jacobian_x * np.power(self.b, self.x.compute()) * np.log(self.b)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class PowNode(OperationNode):
	def __init__(self, x, e):
		self.x = x
		self.e = e
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.power(self.x.compute(), self.e)
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		return jacobian_x * self.e * np.power(self.x.compute(), self.e - 1)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class DivisionNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.divide(self.x.compute(), self.y.compute())
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		jacobian_y = self.y.jacobian(var)
		self.x_computed = self.x.compute()
		self.y_computed = self.y.compute()
		div = np.divide(self.x_computed, np.power(self.y_computed, 2))
		return jacobian_x * np.divide(1, self.y.compute()) - jacobian_y * div

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class MaxNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.maximum(self.x.compute(), self.y.compute())
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		jacobian_y = self.y.jacobian(var)

		condition = self.x.compute() >= self.y.compute()
		return np.where(condition, jacobian_x, jacobian_y)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class MinNode(OperationNode):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.shape = x.shape
		self.computed = None

	def compute(self):
		if self.computed is None:
			self.computed = np.minimum(self.x.compute(), self.y.compute())
		return self.computed

	def jacobian(self, var):
		jacobian_x = self.x.jacobian(var)
		jacobian_y = self.y.jacobian(var)

		condition = self.x.compute() <= self.y.compute()
		return np.where(condition, jacobian_x, jacobian_y)

	def backward(self, var):
		jacobian = self.jacobian(var)
		return jacobian_to_gradient(jacobian, var)


class AvgNode(OperationNode):
	def __init__(self, arr):
		self.arr = arr
		self.shape = arr[0].shape
		self.computed = None

	@staticmethod
	def mean(arr):
		sum = arr[0]
		for a in arr[1:]:
			sum += a
		return sum / len(arr)

	def compute(self):
		if self.computed is None:
			self.computed = AvgNode.mean(self.arr)
		return self.computed

	def jacobian(self, var):
		return AvgNode.mean([a.jacobian(var) for a in self.arr])

	def backward(self, var):
		return AvgNode.mean([a.backward(var) for a in self.arr])

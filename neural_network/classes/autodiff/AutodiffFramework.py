from neural_network.classes.autodiff import Variable
from neural_network.classes.autodiff.TreeNodes import *


class AutodiffFramework:
	def __init__(self, strict: bool):
		self.variables = []
		self.variable_nodes = []
		self.tree_nodes = []
		self.strict = strict

	def reset(self):
		self.tree_nodes = []
		self.variable_nodes = []

	def reset_all(self):
		self.variables = []
		self.reset()

	def add_variable(self, var, name=""):
		if not isinstance(var, np.ndarray):
			raise ValueError("variable can only be created from numpy ndarray")
		v = Variable(var, name)
		if self.strict:
			self.variables.append(v)
		return v

	def register_node(self, node):
		if self.strict:

			if isinstance(node, VariableNode):
				self.variable_nodes.append(node)
			else:
				self.tree_nodes.append(node)
		return node

	def convert_to_node(self, x, shape=None):
		if isinstance(x, Variable):
			if self.strict and x not in self.variables:
				raise ValueError("only variable created with add_variable method can be used when strict mode is used")
			p = list(filter(lambda v: v.variable == x, self.variable_nodes))
			if p:
				return p[0]
			else:
				x_node = VariableNode(x)
				return self.register_node(x_node)
		elif isinstance(x, TreeNode):
			if self.strict and x not in self.tree_nodes:
				raise ValueError("only nodes from current framework can be used when strict mode is used")
			return x
		else:
			if x.shape == () and shape:
				x = x * np.ones(shape=shape)
			x_node = ConstantNode(x)
			return self.register_node(x_node)

	def transpose(self, x):
		x_node = self.convert_to_node(x)

		node = TransposeNode(x_node)
		return self.register_node(node)

	def add(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = AddNode(x_node, y_node)
		return self.register_node(node)

	def sum(self, x, ax):
		x_node = self.convert_to_node(x)

		node = SumNode(x_node, ax)
		return self.register_node(node)

	def sub(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = SubNode(x_node, y_node)
		return self.register_node(node)

	def product(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = ProductNode(x_node, y_node)
		return self.register_node(node)

	def matmul(self, x, y):
		x_node = self.convert_to_node(x)
		y_node = self.convert_to_node(y)

		node = MatMulNode(x_node, y_node)
		return self.register_node(node)

	def get(self, x, i):
		x_node = self.convert_to_node(x)

		node = GetNode(x_node, i)
		return self.register_node(node)

	# def insert(self, x, y, i, ax):
	# 	x_node = self.convert_to_node(x)
	# 	y_node = self.convert_to_node(y)
	#
	# 	node = InsertNode(x_node, y_node, i, ax)
	# 	return self.add_node(node)

	def concat(self, x, y, ax):
		x_node = self.convert_to_node(x)
		y_node = self.convert_to_node(y)

		node = ConcatNode(x_node, y_node, ax)
		return self.register_node(node)

	def stack(self, arr, ax):
		nodes = [self.convert_to_node(x) for x in arr]
		node = StackNode(nodes, ax)
		return self.register_node(node)

	def exp(self, b, x):
		x_node = self.convert_to_node(x)

		node = ExpNode(b, x_node)
		return self.register_node(node)

	def pow(self, x, e):
		x_node = self.convert_to_node(x)

		node = PowNode(x_node, e)
		return self.register_node(node)

	def division(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = DivisionNode(x_node, y_node)
		return self.register_node(node)

	def max(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = MaxNode(x_node, y_node)
		return self.register_node(node)

	def min(self, x, y):
		x_node = self.convert_to_node(x, y.shape)
		y_node = self.convert_to_node(y, x.shape)

		node = MinNode(x_node, y_node)
		return self.register_node(node)

	def avg(self, arr):
		nodes = [self.convert_to_node(x) for x in arr]
		node = AvgNode(nodes)
		return self.register_node(node)

	def gradient(self, root, var):
		if self.strict:
			if root not in self.tree_nodes:
				raise ValueError("only nodes from current framework can be used when strict mode is used")
			if var not in self.variables:
				raise ValueError("only variable created with add_variable method can be used when strict mode is used")
		return root.backward(var)

	def jacobian(self, root, var):
		if self.strict:
			if root not in self.tree_nodes:
				raise ValueError("only nodes from current framework can be used when strict mode is used")
			if var not in self.variables:
				raise ValueError("only variable created with add_variable method can be used when strict mode is used")
		return root.jacobian(var)


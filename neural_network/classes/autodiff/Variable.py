import numpy as np


class Variable:
	def __init__(self, value, name=""):
		self.value = value
		self.shape = value.shape
		if not name:
			self.name = str(self)

	def get_value(self):
		return self.value

	def assign(self, value):
		self.value = value

	def __add__(self, other):
		self.value += other
		return self

	def __sub__(self, other):
		self.value -= other
		return self

	def __mul__(self, other):
		return self.value * other

	def assign_add(self, value):
		self.value += value
		return self

	def assign_sub(self, value):
		self.value -= value
		return self

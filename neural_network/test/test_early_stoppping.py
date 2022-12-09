import unittest
import math
import numpy as np
from neural_network.classes.Validation import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
	def test_plateau(self):
		plateau_function = lambda x: x if x <= 10 else 10

		es = EarlyStopping(
			monitor="monitored_value",
			patience=5,
			mode="max",
			min_delta=0.1,
			restore_best_weight=False
		)
		for x in np.arange(1,20, 1):
			flag = es.add_monitored_value(
				layers=[],
				monitored_values={
					"monitored_value": plateau_function(x)
				}
			)
			self.assertTrue((x < 15 and flag == False) or (x == 15 and flag == True))
			if flag:
				break

	def test_restore_best_weight(self):
		function = lambda x: -abs(x-10) +5

		es_not_restore_weight = EarlyStopping(
			monitor="monitored_value",
			patience=5,
			mode="max",
			min_delta=0.1,
			restore_best_weight=False
		)
		es_restore_weight = EarlyStopping(
			monitor="monitored_value",
			patience=5,
			mode="max",
			min_delta=0.1,
			restore_best_weight=True
		)

		for x in range(0, 30, 1):
			flag1 = es_not_restore_weight.add_monitored_value(
				layers=[x],
				monitored_values={
					"monitored_value": function(x)
				}
			)
			flag2 = es_restore_weight.add_monitored_value(
				layers=[x],
				monitored_values={
					"monitored_value": function(x)
				}
			)

			if flag1 and flag2:
				last_weights = es_not_restore_weight.get_best_weights()
				best_weights = es_restore_weight.get_best_weights()
				self.assertIsInstance(last_weights, list)
				self.assertIsInstance(best_weights, list)
				self.assertEqual(last_weights[0], x)
				self.assertEqual(best_weights[0], 10)
				break

	def test_descent(self):
		function = lambda x: -(3*math.sin(x) + x)

		es = EarlyStopping(
			monitor="monitored_value",
			patience=2,
			mode="min",
			min_delta=0,
			restore_best_weight=False
		)
		for x in [1.7, 1.8, 1.9, 2, 2.1, 2.2]:
			flag = es.add_monitored_value(
				layers=[],
				monitored_values={
					"monitored_value": function(x)
				}
			)
			self.assertTrue((x < 2.2 and flag == False) or (x==2.1 and flag == True))
			if flag:
				break

	def test_double_descent(self):
		function = lambda x: -(3*math.sin(x) + x)

		es = EarlyStopping(
			monitor="monitored_value",
			patience=4,
			mode="max",
			min_delta=0,
			restore_best_weight=False
		)
		for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
			flag = es.add_monitored_value(
				layers=[],
				monitored_values={
					"monitored_value": function(x)
				}
			)
			self.assertTrue((x < 8 and flag == False) or (x==8 and flag == True))
			if flag:
				break

	def test_slow_growth(self):
		function = lambda x: math.log(x)*0.5

		es = EarlyStopping(
			monitor="monitored_value",
			patience=5,
			mode="max",
			min_delta=0.1,
			restore_best_weight=False
		)
		for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
			flag = es.add_monitored_value(
				layers=[],
				monitored_values={
					"monitored_value": function(x)
				}
			)
			self.assertTrue((x < 10 and flag == False) or (x==10 and flag == True))
			if flag:
				break

# coding: utf-8

"""
	定义一个DNN的简单的模版
"""

import numpy as np


class DnnModel(object):
	def __init__(self, data_set):
		self.train_x_set = data_set.get('train_x')
		self.train_y_set = data_set.get('train_y')
		self.test_x_set = data_set.get('test_x')
		self.test_y_set = data_set.get('test_y')
		self.grads = {}
		self.params = {}
		self.cache = {}
		self.cost = []
		self.super_params = {
			'num_times': 2000,
			'learning_rate': 0.005,
		}
		self.pre_do()

	def pre_do(self):
		pass

	def init_param(self):
		pass

	def forward_process(self, x_set):
		pass

	def back_process(self,):
		pass

	def compute_cost(self, a, y):
		m = y.shape[1]
		cost = np.multiply(y, np.log(a)) + np.multiply(1-y, np.log(1-a))
		cost = - np.sum(cost)/m
		# self.cost.append(cost)
		return cost

	def optimize(self, learning_rate=0.001):
		pass

	def predict(self):
		this_test_x = self.test_x_set
		predict_y = self.forward_process(this_test_x)
		return predict_y

	def compute_error(self):
		pass

	def model(self,):
		pass




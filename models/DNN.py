# coding: utf-8

"""
	定义一个DNN模版
"""
import numpy as np

from models.defs import L_MODEL_CONFIG
from models.utils import init_params, l_model_forward, l_model_backward, default_cost, update_params, plot, \
	default_cost_derivative


class DnnModel(object):
	def __init__(self, data_set, neural_conf=L_MODEL_CONFIG):
		self.train_x_set = data_set.get('train_x')
		self.train_y_set = data_set.get('train_y')
		self.test_x_set = data_set.get('test_x')
		self.test_y_set = data_set.get('test_y')

		self.neural_conf = neural_conf

		self.params = dict()
		self.caches = dict()
		self.costs = list()

		self.len_neural = len(self.neural_conf)

		self.hyper_params = {
			'num_iter': 2000,
			'learning_rate': 0.005,
		}
		self.grads = dict()
		self.prepare_do()

	def prepare_do(self):
		pass

	def init_params(self):
		self.params = init_params(self.train_x_set, self.neural_conf, )

	def forward_process(self, x_set):
		self.caches = l_model_forward(x_set, self.params, self.neural_conf)

	def backward_process(self,):
		A = self.caches.get('A')[self.len_neural]
		dAl = self.compute_cost_derivative(A, self.train_y_set)
		self.grads = l_model_backward(self.train_x_set, dAl, self.params, self.caches, self.neural_conf)

	def compute_cost(self, a, y, compute_error=False):
		cost = default_cost(a, y)
		if not compute_error:
			self.costs.append(cost)
		return cost

	def compute_cost_derivative(self, a, y):
		return default_cost_derivative(a, y)

	def update_params(self):
		self.params = update_params(self.params, self.grads, self.neural_conf, self.hyper_params)

	def predict(self):
		self.forward_process(self.test_x_set)
		A = self.caches.get('A')[self.len_neural]
		return self.softmax(A)

	def compute_error(self,):
		A = self.caches.get('A')[self.len_neural]
		train_correct = self.compute_cost(A, self.train_y_set, compute_error=True)
		self.forward_process(self.test_x_set)
		A = self.caches.get('A')[self.len_neural]
		test_correct = self.compute_cost(A, self.test_y_set, compute_error=True)
		print('在训练集的正确率为{}%'.format(train_correct * 100))
		print('在测试集的正确率为{}%'.format(test_correct * 100))
		self.plot('', self.costs)

	def plot(self, x, y):
		plot(x, y)

	def softmax(self, a):
		pass

	def model(self,):
		num_iter = self.hyper_params.get('num_iter')
		self.init_params()
		for i in range(num_iter):
			self.forward_process(self.train_x_set)
			self.backward_process()
			A = self.caches.get('A')[self.len_neural]
			if (1+i) % 100 == 0 or i % 100 == 0:
				cost = self.compute_cost(A, self.train_y_set)
				print('第{}次训练，损失为{}'.format(i + 1, cost))
			self.update_params()
		self.compute_error()



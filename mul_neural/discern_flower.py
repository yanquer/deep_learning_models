# coding: utf-8

"""
		一个两层的浅层神经网络
		其中第一层隐藏神经元个数为4
"""
import numpy as np
from matplotlib import pyplot as plt
from DNN import DnnModel
from mul_neural.planar_utils import load_planar_dataset
from pubfun import tanh, sigmoid


class DiscernFlower(DnnModel):
	def __init__(self, data_set):
		super().__init__(data_set)
		self.super_params['H'] = 4 	# 隐藏层神经元个数
		self.super_params['num_times'] = 10000
		self.super_params['learning_rate'] = 0.5

	def init_param(self):
		X, Y = self.train_x_set, self.train_y_set
		H = self.super_params.get('H')
		self.params = {
			'W1': np.random.randn(H, X.shape[0]) * 0.01,
			'b1': np.zeros((H, 1)),
			'W2': np.random.randn(Y.shape[0], H) * 0.01,
			'b2': np.zeros((Y.shape[0], 1)),
		}

	def forward_process(self, x_set):
		W1, b1 = self.params.get('W1'), self.params.get('b1')
		W2, b2 = self.params.get('W2'), self.params.get('b2')
		assert W1.shape[1] == x_set.shape[0]
		Z1 = np.dot(W1, x_set) + b1
		A1 = tanh(Z1)
		Z2 = np.dot(W2, A1) + b2
		A2 = sigmoid(Z2)
		assert A2.shape == (1, x_set.shape[1])
		self.cache = {
			'A1': A1,
			'A2': A2,
		}

		return A2

	def back_process(self,):
		A1, A2 = self.cache.get('A1'), self.cache.get('A2')
		X, Y = self.train_x_set, self.train_y_set
		W1, W2 = self.params.get('W1'), self.params.get('W2')
		m = Y.shape[1]

		assert A2.shape == Y.shape
		dZ2 = A2 - Y
		dW2 = np.dot(dZ2, A1.T)/m
		db2 = np.sum(dZ2, axis=1, keepdims=True)/m

		dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
		dW1 = np.dot(dZ1, X.T)/m
		db1 = np.sum(dZ1, axis=1, keepdims=True)/m

		self.grads = {
			'dW1': dW1,
			'db1': db1,
			'dW2': dW2,
			'db2': db2,
		}

	def optimize(self, learning_rate=0.001):
		W1, b1 = self.params.get('W1'), self.params.get('b1')
		W2, b2 = self.params.get('W2'), self.params.get('b2')

		dW1, db1 = self.grads.get('dW1'), self.grads.get('db1')
		dW2, db2 = self.grads.get('dW2'), self.grads.get('db2')

		W1 -= learning_rate * dW1
		W2 -= learning_rate * dW2
		b1 -= learning_rate * b1
		b2 -= learning_rate * b2

		self.params = {
			'W1': W1,
			'b1': b1,
			'W2': W2,
			'b2': b2,
		}

	def compute_error(self):
		A2 = self.forward_process(self.train_x_set)
		y_train = np.zeros(A2.shape)
		y_train[A2 > 0.5] = 1

		correct_train = (1 - np.mean(np.abs(y_train - self.train_y_set))) * 100
		print('在训练集上的正确率为：{}%'.format(correct_train))
		self.art()

	def art(self):
		x = np.array([x+1 for x in range(self.super_params.get('num_times')) if (x+1) % 1000 == 0 or x % 1000 == 0])
		y = np.array(self.cost)
		plt.plot(x, y)
		plt.xlabel('num_times')
		plt.ylabel('cost')
		plt.show()

	def model(self,):
		self.init_param()
		num_times = self.super_params.get('num_times')
		for i in range(num_times):
			self.forward_process(self.train_x_set)
			this_cost = self.compute_cost(self.cache.get('A2'), self.train_y_set)
			if (1+i) % 1000 == 0 or i % 1000 == 0:
				print('第{}次训练，损失为{}'.format(i + 1, this_cost))
				self.cost.append(this_cost)
			self.back_process()
			learning_rate = self.super_params.get('learning_rate')
			self.optimize(learning_rate)
		self.compute_error()

if __name__ == '__main__':
	x, y = load_planar_dataset()
	data = {
		'train_x': x,
		'train_y': y,
	}
	mol = DiscernFlower(data)
	mol.model()




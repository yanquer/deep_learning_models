# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt

from DNN import DnnModel
from pubfun import sigmoid
from single_neuron.lr_utils import load_dataset


class DiscernCat(DnnModel):
	def __init__(self, train_set, test_set, data_set):
		super().__init__(data_set)
		self.train_x_set = train_set.get('train_x')
		self.train_y_set = train_set.get('train_y')
		self.test_x_set = test_set.get('test_x')
		self.test_y_set = test_set.get('test_y')

	def pre_do(self):
		this_train_x, this_test_x = self.train_x_set, self.test_x_set
		self.train_x_set = this_train_x.reshape(this_train_x.shape[0], -1).T/255
		self.test_x_set = this_test_x.reshape(this_test_x.shape[0], -1).T/255

	def init_param(self):
		n = self.train_x_set.shape[0]
		self.params = {
			'W': np.zeros((1, n)),
			'b': 0,
		}

	def forward_process(self, x_set):
		vector_w = self.params.get('W')
		vector_b = self.params.get('b')
		assert vector_w.shape[1] == x_set.shape[0]
		Z = np.dot(vector_w, x_set) + vector_b
		A = sigmoid(Z)
		Y = np.zeros(A.shape)
		Y[A > 0.5] = 1
		self.cache = {
			'Z': Z,
			'A': A,
			'Y': Y,
		}
		return Y

	def back_process(self,):
		A, X, y_predict, Y = self.cache.get('A'), self.train_x_set, self.cache.get('Y'), self.train_y_set
		assert X.shape[1] == Y.shape[1]
		m = Y.shape[1]
		dZ = A - Y
		dW = np.dot(dZ, X.T)/m
		db = np.sum(dZ)/m

		self.grads = {
			'dW': dW,
			'db': db,
		}

	def compute_cost(self, a, y):
		m = y.shape[1]
		cost = np.multiply(y, np.log(a)) + np.multiply(1-y, np.log(1-a))
		cost = - np.sum(cost)/m
		# self.cost.append(cost)
		return cost

	def optimize(self, learning_rate=0.001):
		W, b = self.params.get('W'), self.params.get('b')
		dW, db = self.grads.get('dW'), self.grads.get('db')

		W -= learning_rate * dW
		b -= learning_rate * db
		self.params['W'] = W
		self.params['b'] = b

	def predict(self):
		this_test_x = self.test_x_set
		predict_y = self.forward_process(this_test_x)
		return predict_y

	def art(self):
		x = np.array([x+1 for x in range(self.super_params.get('num_times')) if (x+1) % 100 == 0 or x % 100 == 0])
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
			this_cost = self.compute_cost(self.cache.get('A'), self.cache.get('Y'))
			if (1+i) % 100 == 0 or i % 100 == 0:
				print('第{}次训练，损失为{}'.format(i + 1, this_cost))
				self.cost.append(this_cost)
			self.back_process()
			learning_rate = self.super_params.get('learning_rate')
			self.optimize(learning_rate)
		y_train = self.forward_process(self.train_x_set)
		y_test = self.forward_process(self.test_x_set)

		correct_train = (1 - np.mean(np.abs(y_train - self.train_y_set))) * 100
		correct_test = (1 - np.mean(np.abs(y_test - self.test_y_set))) * 100
		print('在训练集上的正确率为：{}%'.format(correct_train))
		print('在测试集上的正确率为：{}%'.format(correct_test))
		self.art()


if __name__ == '__main__':
	train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
	train_set_o = {
		'train_x': train_set_x_orig,
		'train_y': train_set_y_orig,
	}
	test_set_o = {
		'test_x': test_set_x_orig,
		'test_y': test_set_y_orig,
	}
	test_model = DiscernCat(train_set_o, test_set_o)
	test_model.model()


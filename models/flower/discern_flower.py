# coding: utf-8

"""
		一个两层的浅层神经网络
		其中第一层隐藏神经元个数为4
"""
import numpy as np
from matplotlib import pyplot as plt

from models import defs
from models.DNN import DnnModel
from planar_utils import load_planar_dataset


class DiscernFlower(DnnModel):
	def __init__(self, data_set, neural_conf):
		super().__init__(data_set, neural_conf)
		self.hyper_params = {
			'num_iter': 10000,
			'learning_rate': 0.5,
		}

	def compute_error(self):
		A = self.caches.get('A')[self.len_neural]
		y_train = np.round(A)

		correct_train = (1 - np.mean(np.abs(y_train - self.train_y_set))) * 100
		print('在训练集上的正确率为：{}%'.format(correct_train))
		print('准确率: %d' % float(
			(np.dot(self.train_y_set, y_train.T) + np.dot(1 - self.train_y_set, 1 - y_train.T)) / float(self.train_y_set.size) * 100) + '%')

		self.plot('', self.costs)


if __name__ == '__main__':
	x, y = load_planar_dataset()
	data = {
		'train_x': x,
		'train_y': y,
	}
	data_conf = (
		(4, defs.TANH,),
		(1, defs.SIGMOID,),
	)
	mol = DiscernFlower(data, data_conf)
	mol.model()




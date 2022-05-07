# coding: utf-8
import numpy as np

from models import defs
from models.DNN import DnnModel
from old_models.single_neuron.lr_utils import load_dataset


class DiscernCat(DnnModel):
	def __init__(self, data_set, neural_conf):
		super().__init__(data_set, neural_conf)

	def prepare_do(self):
		this_train_x, this_test_x = self.train_x_set, self.test_x_set
		self.train_x_set = this_train_x.reshape(this_train_x.shape[0], -1).T/255
		self.test_x_set = this_test_x.reshape(this_test_x.shape[0], -1).T/255

	def compute_error(self,):
		A = self.caches.get('A')[self.len_neural]
		y_train = self.softmax(A)
		train_correct = (1 - np.mean(np.abs(y_train - self.train_y_set))) * 100
		self.forward_process(self.test_x_set)
		A = self.caches.get('A')[self.len_neural]
		y_test = self.softmax(A)
		test_correct = (1 - np.mean(np.abs(y_test - self.test_y_set))) * 100
		print('在训练集的正确率为{}%'.format(train_correct))
		print('在测试集的正确率为{}%'.format(test_correct))
		self.plot('', self.costs)

	def softmax(self, a):
		y_predict = np.zeros(a.shape)
		y_predict[a > 0.5] = 1
		return y_predict


if __name__ == '__main__':
	train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
	data = {
		'train_x': train_set_x_orig,
		'train_y': train_set_y_orig,
		'test_x': test_set_x_orig,
		'test_y': test_set_y_orig,
	}
	data_conf = (
		(1, defs.SIGMOID, ),
	)

	test_model = DiscernCat(data, data_conf)
	test_model.model()


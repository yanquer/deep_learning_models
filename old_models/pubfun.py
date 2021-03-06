# coding: utf-8

import numpy as np


def tanh(z):
	"""
		tanh 函数
	:param z:
	:return:
	"""
	return np.tanh(z)


def tanh_derivative(z):
	"""
		tanh 求导
	:param z:
	:return:
	"""
	return 1 - np.power(tanh(z), 2)


def sigmoid(z):
	"""
		sigmoid 函数
	:param z:
	:return:
	"""
	return 1/(1+np.exp(-z))


def sigmoid_derivative(z):
	"""
		sigmoid 求导
	:param z:
	:return:
	"""
	return sigmoid(z) * (1 - sigmoid(z))


def wt_func(mat_w, mat_x, b):
	"""
		Z = W.T * X + b
	:param mat_w:
	:param mat_x:
	:param b:
	:return:
	"""
	assert mat_w.shape[0] == mat_x.shape[0]
	return np.dot(mat_w.T, mat_x) + b





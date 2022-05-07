# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

from models import defs


def sigmoid(x):
	return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))


def tanh_derivative(x):
	return 1 - np.tanh(x) ** 2


def relu(x):
	# return max(0, x)
	x[x < 0] = 0
	return x


def relu_derivative(x):
	return 1


def leaky_relu(x, rate=0.01):
	return max(rate*x, x)


def init_params(X, neural_model_conf, init_rate=0.01):
	num_l_prev = X.shape[0]
	model_len = len(neural_model_conf)
	params = {
		'W': [0 for _ in range(model_len+1)],
		'b': [0 for _ in range(model_len+1)],
	}
	for l in range(model_len):
		num_l = neural_model_conf[l][0]
		params['W'][l+1] = np.random.randn(num_l, num_l_prev) * init_rate
		params['b'][l+1] = np.zeros((num_l, 1))
		num_l_prev = num_l
	return params


def default_cost(A, Y, ):
	m = Y.shape[1]
	cost = np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T)
	cost = - np.sum(cost)/m
	return cost


def default_cost_derivative(A, Y):
	assert A.shape == Y.shape
	r1 = (1-Y)/(1-A) - Y/A
	r2 = - Y/A + (1-Y)/(1-A)
	# d_str4 = ('A\n{}\n'.format(A))
	# d_str5 = ('Y\n{}\n'.format(Y))
	# d_str1 = ('r1 == r2 \n{}\n'.format(r1 == r2))
	# d_str2 = ('r1\n{}\n'.format(r1))
	# d_str3 = ('r2\n{}\n'.format(r2))
	# write(d_str4 + d_str5 + d_str1 + d_str2 + d_str3)
	return (1-Y)/(1-A) - Y/A


ACTIVE_FUN_DICT = {
	defs.SIGMOID: sigmoid,
	defs.TANH: np.tanh,
	defs.RELU: relu,
	defs.LEAKY_RELU: leaky_relu,
}

ACTIVE_FUN_DERIVATIVE_DICT = {
	defs.SIGMOID: sigmoid_derivative,
	defs.TANH: tanh_derivative,
	defs.RELU: relu_derivative,
	defs.LEAKY_RELU: None,
}


def l_model_forward(X, params, neural_model_conf):
	model_len = len(neural_model_conf)
	caches = {'Z': [0 for _ in range(model_len+1)], 'A': [0 for _ in range(model_len+1)], }
	Al = X
	w_list, b_list = params.get('W'), params.get('b')
	for l in range(1, model_len+1):
		W = w_list[l]
		b = b_list[l]
		assert W.shape[1] == Al.shape[0]
		Zl = np.dot(W, Al) + b

		l_fun_name = neural_model_conf[l-1][1]
		l_fun = ACTIVE_FUN_DICT.get(l_fun_name)
		assert l_fun
		Al = l_fun(Zl)

		caches['Z'][l] = Zl
		caches['A'][l] = Al
	return caches


def l_model_backward(X, dAl, params: dict, caches: dict, neural_model_conf, ):
	m = X.shape[1]
	model_len = len(neural_model_conf)
	grads = {
		'dW': [0 for _ in range(model_len + 1)],
		'db': [0 for _ in range(model_len + 1)],
	}
	assert dAl.any()

	w_list, b_list = params.get('W'), params.get('b')
	a_list, z_list = caches.get('A'), caches.get('Z')

	for l in range(model_len, 0, -1):
		l_fun_name = neural_model_conf[l-1][1]
		l_fun_der = ACTIVE_FUN_DERIVATIVE_DICT.get(l_fun_name)
		assert l_fun_der
		Zl = z_list[l]
		dZl = dAl * l_fun_der(Zl)
		Al_p = a_list[l-1] if l != 1 else X
		assert dZl.shape[1] == Al_p.shape[1]
		dWl = np.dot(dZl, Al_p.T)/m
		dbl = np.sum(dZl, axis=1, keepdims=True)/m
		Wl = w_list[l]
		assert dZl.shape[0] == Wl.shape[0]
		if l != 1:
			dAl = np.dot(Wl.T, dZl)

		grads['dW'][l] = dWl
		grads['db'][l] = dbl
	return grads


def update_params(params: dict, grads: dict, neural_model_conf, hyper_params: dict):
	dw_list, db_list = grads.get('dW'), grads.get('db')
	model_len = len(neural_model_conf)
	learning_rate = hyper_params.get('learning_rate')
	assert learning_rate

	for l in range(1, model_len+1):
		params['W'][l] -= learning_rate * dw_list[l]
		params['b'][l] -= learning_rate * db_list[l]

	return params


def plot(x, y, x_label='num_iter', y_label='cost'):
	plt.plot(y)
	plt.xlabel(xlabel=x_label)
	plt.ylabel(ylabel=y_label)
	plt.show()


def write(data_str, file='log.txt'):
	with open(file, 'w') as f:
		f.writelines(data_str)

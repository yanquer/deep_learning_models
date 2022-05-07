# coding: utf-8


SIGMOID = 'sigmoid'
TANH = 'tanh'
RELU = 'relu'
LEAKY_RELU = 'leaky_relu'

# 配置神经网络模型，层数以及每一层神经元个数 and 激活函数
L_MODEL_CONFIG = (
	# (当前层神经原个数),
	(4, RELU, ),
	(3, RELU, ),
	(1, SIGMOID, ),
)





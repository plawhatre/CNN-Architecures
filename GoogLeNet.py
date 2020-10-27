import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from colorama import init
from termcolor import *
init()

# InceptionNet_v1 components
class Dense(tf.keras.layers.Layer):
	def __init__(self, mi, mo, f= lambda x: x, disp_name=True):
		super(Dense, self).__init__()
		self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float64))
		self.b = tf.Variable(np.zeros(mo, dtype=np.float64))
		self.f = f
		# self.name = 'Dense'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) + self.b)

class Flatten:
	def __init__(self, disp_name=True):
		self.name = 'Flatten'
		if disp_name == True:
			print('\tAdding',self.name)

	@staticmethod
	def forward(X):
		return tf.reshape(X, [tf.shape(X)[0], -1])

class MaxPoolLayer:
	def __init__(self, dim, stride, disp_name=True):
		self.dim = dim
		self.stride = stride
		self.name = 'MaxPoolLayer'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.max_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, self.stride, self.stride, 1],
			padding='SAME')

class AvgPoolLayer:
	def __init__(self, dim, stride, disp_name=True):
		self.dim = dim
		self.stride = stride
		self.name = 'AvgPoolLayer'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.avg_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, self.stride, self.stride, 1],
			padding='SAME')

class GlobalAvgPoolLayer:
	def __init__(self, disp_name=True):
		self.name = 'GlobalAvgPoolLayer'
		if disp_name == True:
			print('\tAdding', self.name)

	def forward(self, X):
		return tf.reduce_mean(X, axis=[1, 2])

class Dropout:
	def __init__(self, rate, disp_name=True):
		self.rate =rate
		self.name = 'Dropout'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.dropout(X, self.rate)

def init_filter(d, mi, mo, stride):
	return (np.random.randn(d, d, mi, mo) * np.sqrt(2.0 / (d * d * mi)))

class ConvLayer(tf.keras.layers.Layer):
	def __init__(self, d, mi, mo, stride=1, padding='SAME', disp_name=True):
		super(ConvLayer, self).__init__()
		self.W = tf.Variable(init_filter(d, mi, mo, stride))
		self.b = tf.Variable(np.zeros(mo, dtype=np.float64))
		self.stride = stride
		self.padding = padding
		# self.name = 'ConvLayer'
		if disp_name==True:
			print('\tAdding',self.name)

	def forward(self, x):
		x = tf.nn.conv2d(x, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
		x = x + self.b
		return tf.nn.relu(x)

class BatchNormLayer(tf.keras.layers.Layer):
	def __init__(self, D, disp_name=True):
		super(BatchNormLayer, self).__init__()
		self.mean = tf.Variable(np.zeros(D, dtype=np.float64), trainable=False)
		self.var = tf.Variable(np.ones(D, dtype=np.float64), trainable=False)
		self.beta = tf.Variable(np.zeros(D, dtype=np.float64))
		self.gamma = tf.Variable(np.ones(D, dtype=np.float64))
		# self.name = 'BatchNormLayer'
		if disp_name==True:
			print('\tAdding',self.name)

	def forward(self, x):
		return tf.nn.batch_normalization(
			x,
			self.mean,
			self.var,
			self.beta,
			self.gamma,
			0.001)

class InceptionLayer(tf.keras.layers.Layer):
	def __init__(self, mi, mo_1x1, mo_3x3_reduce, mo_3x3, mo_5x5_reduce, mo_5x5, mo_pool_proj):
		super(InceptionLayer, self).__init__()
		self.p1_layers = [
		ConvLayer(1, mi, mo_1x1, stride=1, padding='SAME', disp_name=False)
		]
		self.p2_layers = [
		ConvLayer(1, mi, mo_3x3_reduce, stride=1, padding='SAME', disp_name=False),
		ConvLayer(3, mo_3x3_reduce, mo_3x3, stride=1, padding='SAME', disp_name=False)
		]
		self.p3_layers = [
		ConvLayer(1, mi, mo_5x5_reduce, stride=1, padding='SAME', disp_name=False),
		ConvLayer(5, mo_5x5_reduce, mo_5x5, stride=1, padding='SAME', disp_name=False)
		]
		self.p4_layers = [
		MaxPoolLayer(3, 1, disp_name=False),
		ConvLayer(1, mi, mo_pool_proj, stride=1, padding='SAME', disp_name=False)
		]
		self.branches = [self.p1_layers, self.p2_layers, self.p3_layers, self.p4_layers]

		print('\tAdding',self.name)

	def forward(self, X):
		flag = True
		for branch in self.branches:
			Xin = copy.deepcopy(X)
			for layer in branch:
				Xin = layer.forward(Xin)
			if flag == True:
				flag = False
				Xout = Xin
			else:
				Xout = tf.concat([Xout, Xin], -1)

		return Xout

# InceptionNet_v1 
class InceptionNet_v1(tf.keras.models.Model):
	def __init__(self, n_classes):
		super(InceptionNet_v1, self).__init__()
		cprint('Initializing InceptionNet_v1............', 'green')

		self.main_branch1 = [
		# Conv-MaxPool 
		ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME', disp_name=True),
		MaxPoolLayer(dim=3, stride=2, disp_name=True),
		# Conv-2x
		ConvLayer(d=1, mi=64, mo=64, stride=1, padding='SAME', disp_name=True),
		ConvLayer(d=3, mi=64, mo=192, stride=1, padding='SAME', disp_name=True),
		# MaxPool 
		MaxPoolLayer(dim=3, stride=2, disp_name=True),
		# Inceptionlayer-2x
		InceptionLayer(mi=192, mo_1x1=64,
			mo_3x3_reduce=96,
			mo_3x3=128,
			mo_5x5_reduce=16,
			mo_5x5=32,
			mo_pool_proj=32),
		InceptionLayer(mi=256, mo_1x1=128,
			mo_3x3_reduce=128,
			mo_3x3=192,
			mo_5x5_reduce=32,
			mo_5x5=96,
			mo_pool_proj=64),
		# MaxPool 
		MaxPoolLayer(dim=3, stride=2, disp_name=True),
		# Inceptionlayer
		InceptionLayer(mi=480, mo_1x1=192,
			mo_3x3_reduce=96,
			mo_3x3=208,
			mo_5x5_reduce=16,
			mo_5x5=48,
			mo_pool_proj=64)
		]

		self.auxilliary_branch1 = [
		# AvgPool
		AvgPoolLayer(dim=5, stride=3, disp_name=False), 
		# Conv
		ConvLayer(d=1, mi=512, mo=128, disp_name=False),
		# Flatten
		Flatten(disp_name=False),
		# Dense
		Dense(mi=128, mo=1024, f=tf.nn.relu, disp_name=False),
		# Dropout
		Dropout(0.7, disp_name=False),
		#Final side layer
		Dense(mi=1024, mo=n_classes, disp_name=False)
		]

		self.main_branch2 = [
		# Inceptionlayer-3x
		InceptionLayer(mi=512, mo_1x1=160,
			mo_3x3_reduce=112,
			mo_3x3=224,
			mo_5x5_reduce=24,
			mo_5x5=64,
			mo_pool_proj=64),
		InceptionLayer(mi=512, mo_1x1=128,
			mo_3x3_reduce=128,
			mo_3x3=256,
			mo_5x5_reduce=24,
			mo_5x5=64,
			mo_pool_proj=64),
		InceptionLayer(mi=512, mo_1x1=112,
			mo_3x3_reduce=144,
			mo_3x3=288,
			mo_5x5_reduce=32,
			mo_5x5=64,
			mo_pool_proj=64)
		]

		self.auxilliary_branch2 = [
		# AvgPool
		AvgPoolLayer(dim=5, stride=3, disp_name=False), 
		# Conv
		ConvLayer(d=1, mi=528, mo=128, disp_name=False),
		# Flatten
		Flatten(disp_name=False),
		# Dense
		Dense(mi=128, mo=1024, f=tf.nn.relu, disp_name=False),
		# Dropout
		Dropout(0.7, disp_name=False),
		#Final side layer
		Dense(mi=1024, mo=n_classes, disp_name=False)
		]

		self.main_branch3 = [
		# Inceptionlayer
		InceptionLayer(mi=528, mo_1x1=256,
			mo_3x3_reduce=160,
			mo_3x3=320,
			mo_5x5_reduce=32,
			mo_5x5=128,
			mo_pool_proj=128),
		# MaxPool
		MaxPoolLayer(dim=3, stride=2, disp_name=True),
		# Inceptionlayer-2x
		InceptionLayer(mi=832, mo_1x1=256,
			mo_3x3_reduce=160,
			mo_3x3=320,
			mo_5x5_reduce=32,
			mo_5x5=128,
			mo_pool_proj=128),
		InceptionLayer(mi=832, mo_1x1=384,
			mo_3x3_reduce=192,
			mo_3x3=384,
			mo_5x5_reduce=48,
			mo_5x5=128,
			mo_pool_proj=128),
		GlobalAvgPoolLayer(disp_name=True),
		Dropout(0.4),
		Dense(mi=1024, mo=n_classes, disp_name=True)
		]

	def forward(self, X):
		# Branches: Main and Auxilliary
		Xm = copy.deepcopy(X)
		for layer in self.main_branch1:
			Xm = layer.forward(Xm)
		
		X1 = copy.deepcopy(Xm)
		for layer in self.auxilliary_branch1:
			X1 = layer.forward(X1)

		for layer in self.main_branch2:
			Xm = layer.forward(Xm)

		X2 = copy.deepcopy(Xm)
		for layer in self.auxilliary_branch2:
			X2 = layer.forward(X2)

		for layer in self.main_branch3:
			Xm = layer.forward(Xm)

		return X1, X2, Xm

	# Cost
	def cost(self, Y, logits):
		loss = tf.math.reduce_sum(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
		return loss

	def gradient_update(self, X, Y, optimizer):
		with tf.GradientTape() as t:
			Y_hat = self.forward(X)
			auxilliary_loss1 = self.cost(Y, Y_hat[0])
			auxilliary_loss2 = self.cost(Y, Y_hat[1])
			main_loss = self.cost(Y, Y_hat[2])
			Loss = 0.3 * auxilliary_loss1 + 0.3 * auxilliary_loss2 + main_loss

		grads = t.gradient(Loss, self.trainable_weights)
		optimizer.apply_gradients(zip(grads, self.trainable_weights))
		return Loss

	def fit(self, X, Y, epochs=20, batch_size=128, lr=0.001):
		N = X.shape[0]
		n_batches = N // batch_size
		cprint('Train model........','green')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		for i in range(epochs):
			for j in range(n_batches):
				Loss = self.gradient_update(
					X[(j*batch_size):((j+1)*batch_size)], 
					Y[(j*batch_size):((j+1)*batch_size)], 
					optimizer)
				cost_lst.append(Loss/batch_size)
				if j % 10 ==0:
					cprint(f'Epoch: {i+1}, Batch: {j}, Loss: {Loss}', 'red')
		return cost_lst

	def predict(self, X):
		cprint('Making Predictions........','blue')
		_, _, logits = self.forward(X)
		labels = tf.nn.sigmoid(logits)
		return (tf.argmax(labels, axis=1)).numpy()

if __name__ == '__main__':
	# load data
	data = tf.keras.datasets.cifar10
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train, x_test = x_train / 255, x_test / 255
	N_train, H, W, _ = np.shape(x_train)
	N_test, H, W, _ = np.shape(x_test)
	y_train = tf.one_hot(y_train.flatten(), depth=10, dtype=tf.float64)

	#Initilize Model
	model = InceptionNet_v1(n_classes=10)

	# Train the model
	cost_list = model.fit(x_train, y_train)

	# Predict 
	y_pred = model.predict(x_test)

	plt.figure
	plt.plot(cost_list)
	plt.title('Loss Curve')
	plt.show()

	# classification report
	print(classification_report(y_test, y_pred))
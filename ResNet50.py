import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from colorama import init
from termcolor import *
init()

# ResNet50 components
class Dense(tf.keras.layers.Layer):
	def __init__(self, mi, mo):
		super(Dense, self).__init__()
		self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float64))
		self.b = tf.Variable(np.zeros(mo, dtype=np.float64))
		# self.name = 'Dense'
		print('\tAdding',self.name)

	def forward(self, X):
		return tf.matmul(X, self.W) + self.b

class MaxPoolLayer:
	def __init__(self, dim):
		self.dim = dim	
		self.name = 'MaxPoolLayer'
		print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.max_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, 2, 2, 1],
			padding='VALID')

class AvgPoolLayer:
	def __init__(self, ksize):
		self.ksize = ksize
		self.name = 'AvgPoolLayer'
		print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.avg_pool(
			X,
			ksize=[1, self.ksize, self.ksize, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')

class Flatten:
	def __init__(self):
		self.name = 'Flatten'
		print('\tAdding',self.name)

	@staticmethod
	def forward(X):
		return tf.reshape(X, [tf.shape(X)[0], -1])

def init_filter(d, mi, mo, stride):
	return (np.random.randn(d, d, mi, mo) * np.sqrt(2.0 / (d * d * mi)))

class ReLU:
	def __init__(self, disp_name=True):
		self.name = 'ReLU'
		if disp_name==True:
			print('\tAdding',self.name)

	@staticmethod
	def forward(X):
		return tf.nn.relu(X)

class ConvLayer(tf.keras.layers.Layer):
	def __init__(self, d, mi, mo, stride=2, padding='VALID', disp_name=True):
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
		return x

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

class ConvBlock(tf.keras.layers.Layer):
	def __init__(self, mi, f_sizes, stride=2):
		super(ConvBlock, self).__init__()
		self.p1_layers = [ConvLayer(1, mi, f_sizes[0], stride, disp_name=False),
		BatchNormLayer(f_sizes[0], disp_name=False),
		ConvLayer(3, f_sizes[0], f_sizes[1], 1, 'SAME', disp_name=False),
		BatchNormLayer(f_sizes[1], disp_name=False),
		ConvLayer(1, f_sizes[1], f_sizes[2], 1, disp_name=False),
		BatchNormLayer(f_sizes[2], disp_name=False)
		]

		self.p2_layers = [ConvLayer(1, mi, f_sizes[2], stride, disp_name=False),
		BatchNormLayer(f_sizes[2], disp_name=False)
		]
		# self.name = 'ConvBlock'
		print('\tAdding',self.name)

	def forward(self, x):
		xm = copy.deepcopy(x)
		xs = copy.deepcopy(x)

		for layer in self.p1_layers:
			xm = layer.forward(xm)
		
		for layer in self.p2_layers:
			xs = layer.forward(xs)

		return ReLU.forward(xm + xs)

class IdentityBlock(tf.keras.layers.Layer):
	def __init__(self, mi, f_sizes):
		super(IdentityBlock, self).__init__()
		self.layers =[
		ConvLayer(1, mi, f_sizes[0], 1, disp_name=False),
		BatchNormLayer(f_sizes[0], disp_name=False),
		ReLU(disp_name=False),
		ConvLayer(3, f_sizes[0], f_sizes[1], 1, 'SAME', disp_name=False),
		BatchNormLayer(f_sizes[1], disp_name=False),
		ReLU(disp_name=False),
		ConvLayer(1, f_sizes[1], f_sizes[2], 1, disp_name=False),
		BatchNormLayer(f_sizes[2], disp_name=False)]
		# self.name = 'IdentityBlock'
		print('\tAdding',self.name)

	def forward(self, X):
		Xin = copy.deepcopy(X)
		for layer in self.layers:
			Xin = layer.forward(Xin)
		return ReLU.forward(X + Xin)

# ResNet50 
class ResNet50(tf.keras.layers.Layer):
	def __init__(self, output_dims):
		super(ResNet50, self).__init__()
		cprint('Initializing ResNet50............', 'green')
		self.layers = [
		# Before Conv Block
		ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
		BatchNormLayer(64),
		ReLU(),
		MaxPoolLayer(dim=3),
		# Conv Block
		ConvBlock(mi=64, f_sizes=[64, 64, 256], stride=1),
		# Indentity Block: 2x
		IdentityBlock(mi=256, f_sizes=[64, 64, 256]),
		IdentityBlock(mi=256, f_sizes=[64, 64, 256]),
		# Conv Block
		ConvBlock(mi=256, f_sizes=[128, 128, 512], stride=2),
		# Identity Block: 3x
		IdentityBlock(mi=512, f_sizes=[128, 128, 512]), 
		IdentityBlock(mi=512, f_sizes=[128, 128, 512]), 
		IdentityBlock(mi=512, f_sizes=[128, 128, 512]), 
		# Conv Block
		ConvBlock(mi=512, f_sizes=[256, 256, 1024], stride=2),
		# Identity Block: 5x
		IdentityBlock(mi=1024, f_sizes=[256, 256, 1024]),
		IdentityBlock(mi=1024, f_sizes=[256, 256, 1024]),
		IdentityBlock(mi=1024, f_sizes=[256, 256, 1024]),
		IdentityBlock(mi=1024, f_sizes=[256, 256, 1024]),
		IdentityBlock(mi=1024, f_sizes=[256, 256, 1024]),
		# Conv Block
		ConvBlock(mi=1024, f_sizes=[512, 512, 2048], stride=2),
		#Identity Block: 2x
		IdentityBlock(mi=2048, f_sizes=[512, 512, 2048]),
		IdentityBlock(mi=2048, f_sizes=[512, 512, 2048]),
		# Pool--Flatten--Dense
		AvgPoolLayer(ksize=1),
		Flatten(),
		Dense(mi=2048, mo=1000),
		Dense(mi=1000, mo=output_dims)
		]
		cprint('Initialization Complete !!!', 'green')

	def forward(self, X):
		Xin = copy.deepcopy(X)
		for layer in self.layers:
			Xin = layer.forward(Xin)
		return Xin

	# Cost
	def cost(self, Y, logits):
		loss = tf.math.reduce_sum(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
		return loss

	def gradient_update(self, X, Y, optimizer):
		with tf.GradientTape() as t:
			Loss = self.cost(Y, self.forward(X))

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
		logits = self.forward(X)
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
	model = ResNet50(output_dims=10)

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

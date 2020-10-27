import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from colorama import init
from termcolor import *
init()

# Highway Network components
class DenseLayer(tf.keras.layers.Layer):
	def __init__(self, mi, mo, f=lambda x: x):
		super(DenseLayer, self).__init__()
		self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float64))
		self.b = tf.Variable(np.zeros(mo, dtype=np.float64))
		self.f = f
		cprint(f'\tAdding {self.name}')

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) + self.b)

class MaxPoolLayer:
	def __init__(self, dim, stride, padding='VALID', disp_name=True):
		self.dim = dim
		self.stride = stride
		self.padding = padding
		self.name = 'MaxPoolLayer'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.max_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, self.stride, self.stride, 1],
			padding=self.padding)

class GlobalAvgPoolLayer:
	def __init__(self, disp_name=True):
		self.name = 'GlobalAvgPoolLayer'
		if disp_name == True:
			print('\tAdding', self.name)

	def forward(self, X):
		return tf.reduce_mean(X, axis=[1, 2])

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

class HighwayLayer(tf.keras.layers.Layer):
	def __init__(self, d, m_io, bias_init,stride=1, padding='SAME', f=lambda x: x, disp_name=True):
		super(HighwayLayer, self).__init__()
		self.W_h = tf.Variable(init_filter(d, m_io, m_io, stride))
		self.b_h = tf.Variable(np.zeros(m_io, dtype=np.float64)- bias_init)
		self.W_t = tf.Variable(init_filter(d, m_io, m_io, stride))
		self.b_t = tf.Variable(np.zeros(m_io, dtype=np.float64)- bias_init)
		self.stride = stride
		self.padding = padding
		self.f = f
		if disp_name == True:
			print(f'\tAdding {self.name}')

	def forward(self, X):
		T = self.f(tf.nn.conv2d(
			X, self.W_t, strides=[1, self.stride, self.stride, 1], padding=self.padding) + self.b_t)
		H = self.f(tf.nn.conv2d(
			X, self.W_h, strides=[1, self.stride, self.stride, 1], padding=self.padding) + self.b_h)		
		C = tf.math.subtract(1, T)
		
		return tf.math.multiply(T, H) + tf.math.multiply(X, C)

class Flatten:
	def __init__(self):
		self.name = 'Flatten'
		print('\tAdding',self.name)

	@staticmethod
	def forward(X):
		return tf.reshape(X, [tf.shape(X)[0], -1])

# Highway Network: CNN
class HighwayNet_CNN(tf.keras.layers.Layer):
	def __init__(self, highway_dims, n_classes):
		super(HighwayNet_CNN, self).__init__()
		cprint('Initializing HighwayNet_CNN............', 'red')
		self.layers = [
		ConvLayer(d=5, mi=1, mo=highway_dims),
		MaxPoolLayer(3, 2)
		]

		for i in range(2):
			self.layers.append(HighwayLayer(d=3, m_io=highway_dims, bias_init=-1.0))
			self.layers.append(MaxPoolLayer(3, 2))

		self.layers.append(GlobalAvgPoolLayer())
		self.layers.append(Flatten())
		self.layers.append(DenseLayer(mi=highway_dims, mo=10))

		cprint('Initialization Complete!!!', 'red')

	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)

		return X

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

	def fit(self, X, Y, epochs=20, batch_size=256, lr=0.001):
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

if __name__ == "__main__":
	# load data
	data = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train, x_test = x_train / 255, x_test / 255
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = tf.expand_dims(x_train, axis=-1)
	x_test = tf.expand_dims(x_test, axis=-1)
	y_train = tf.one_hot(y_train.flatten(), depth=10, dtype=tf.float64)

	#Initilize Model
	model = HighwayNet_CNN(highway_dims=32, n_classes=10)

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
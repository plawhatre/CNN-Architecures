import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from colorama import init
from termcolor import *
init()

'''
I have deliberated skipped the drop path regularizaion
'''

# FractalNet components
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

def init_filter(d, mi, mo, stride):
	return (np.random.randn(d, d, mi, mo) * np.sqrt(2.0 / (d * d * mi)))

class ConvLayer(tf.keras.layers.Layer):
	def __init__(self, d, mi, mo, stride=1, padding='SAME', disp_name=False):
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

class FractalBlock(tf.keras.layers.Layer):
	def __init__(self, n, d, mi, mo, disp_name=False):
		super(FractalBlock, self).__init__()
		self.n = n
		if n == 1:
			self.layer = ConvLayer(d, mi, mo)
			# print('\t'*self.n+f"On level {self.n} adding on left {self.layer.name}")
		else:
			self.l_layer = ConvLayer(d, mi, mo)
			# print('\t'*self.n+f"On level {self.n} adding on left {self.l_layer.name}")
			self.r_layer = [FractalBlock(n -1, d, mi, mi), FractalBlock(n -1, d, mi, mo)]

		if disp_name==True:
			print('\tAdding',self.name)

	def forward(self, X):
		if self.n == 1:
			Xout = self.layer.forward(X)
		else:
			l_output = self.l_layer.forward(X)


			r_output = copy.deepcopy(X)
			for block in self.r_layer:
				r_output = block.forward(r_output)

			Xout = 0.5*tf.math.add(l_output, r_output)

		return Xout

# FractalNet
class FractalNet(tf.keras.layers.Layer):
	def __init__(self, n, n_classes, f_maps):
		super(FractalNet, self).__init__()

		cprint('Initializing FractalNet............', 'green')
		self.layers = [
		FractalBlock(n, d=3, mi=3, mo=f_maps[0], disp_name=True),
		MaxPoolLayer(dim=3, stride=2),

		FractalBlock(n, d=3, mi=f_maps[0], mo=f_maps[1], disp_name=True),
		MaxPoolLayer(dim=3, stride=1),

		FractalBlock(n, d=3, mi=f_maps[1], mo=f_maps[2], disp_name=True),
		MaxPoolLayer(dim=2, stride=2),

		FractalBlock(n, d=3, mi=f_maps[2], mo=f_maps[3], disp_name=True),
		MaxPoolLayer(dim=2, stride=2),

		FractalBlock(n, d=3, mi=f_maps[3], mo=f_maps[4], disp_name=True),
		MaxPoolLayer(dim=3, stride=1),

		Flatten(),
		Dense(mi=f_maps[4], mo=n_classes)
		]
		
		cprint('Initialization Complete !!!', 'green')

	def forward(self, X):
		Xout = copy.deepcopy(X)
		for layer in self.layers:
			Xout = layer.forward(Xout)

		return Xout

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

	def fit(self, X, Y, epochs=50, batch_size=128, lr=0.001):
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
	model = FractalNet(n=5, n_classes=10, f_maps=[16, 32, 64, 128, 256])


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
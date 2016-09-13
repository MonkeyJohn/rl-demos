import tflearn
import numpy as np

class Model:
	def __init__(self, inp_dim, out_dim, lr=1e-4, sample_size=100):
		"""not defined """
		pass

	def forward(self, x):
		pass

	def backward(self):
		pass

	def record(self, x, a, r):
		pass


class FCNet(Model):
	def __init__(self, inp_dim, num_hidden, out_dim, lr=1e-4, decay_rate=0.99):
		input_layer = tflearn.input_data(shape=[None, inp_dim])
		dense = tflearn.fully_connected(input_layer, num_hidden, activation='relu', name='hidden_layer')
		softmax = tflearn.fully_connected(dense, out_dim, activation='softmax')

		rmsprop = tflearn.RMSProp(learning_rate=lr, decay=decay_rate)
		network = tflearn.regression(softmax, optimizer=rmsprop, loss='categorical_crossentropy')

		self.model = tflearn.DNN(network)
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.memory = []

	def forward(self, x):
		return self.model.predict(x)

	def record(self, x, action, reward):
		self.memory.append(x, action, reward)

	def backward(self):
		N = len(self.memory)
		inp_D = self.inp_dim
		out_D = self.out_dim
		X = np.zeros((N, inp_D))
		y = np.zeros((N, out_D))

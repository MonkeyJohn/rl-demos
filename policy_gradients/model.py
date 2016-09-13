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

	def update(self):
		pass


class FCNet(Model):
	def __init__(self, inp_dim, num_hidden, out_dim, lr=1e-4, decay_rate=0.99, sample_size=100):
		input_layer = tflearn.input_data(shape=[None, inp_dim])
		dense = tflearn.fully_connected(input_layer, num_hidden, activation='relu', name='hidden_layer')
		softmax = tflearn.fully_connected(dense, out_dim, activation='softmax')

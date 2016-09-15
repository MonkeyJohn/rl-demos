import tflearn
import numpy as np
import tensorflow as tf

SUMMARY_DIR = "logs/"
MODEL_DIR = "models/"

class FCNet:
	def __init__(self, inp_dim, out_dim, lr=1e-2, decay_rate=0.99):
		self.sess = tf.Session()
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.lr = lr
		self.decay_rate = decay_rate
		self.X = []
		self.ys = []
		self.actions = []
		self.rs = []

		self.input_layer, self.action_probs = self.create_network()
		self.network_params = tf.trainable_variables()
		
		self.y_fake_label = tf.placeholder(tf.float32, shape=[None, out_dim], name='fake_label')
		#self.y_fake_label = tf.placeholder(tf.float32, shape=[None, 1], name='fake_label')
		self.reward_signal = tf.placeholder(tf.float32, name='reward_signal')

		#self.loss = -tf.reduce_mean(tf.log(self.y_fake_label - self.action_probs) * self.reward_signal)
		"""self.logprobs = tf.log(tf.reduce_sum(tf.mul(self.y_fake_label, self.action_probs), 1))
		self.loss = -tf.reduce_mean(self.logprobs*tf.squeeze(self.reward_signal))
		#self.loss = -tf.reduce_mean(self.logprobs*self.reward_signal)"""

		logprobs = tf.nn.log_softmax(softmax_inputs)
		action_probs = tf.exp(logprobs)
		logp = tf.reduce_sum(logprobs * y_label, [1])  # y_label is at one-hot representation here
		loss = - tf.reduce_mean(logp * advantage)
		self.gradients = tf.gradients(self.loss, self.network_params)

		
		self.grads_buffer_ph = [tf.placeholder(tf.float32) for i in range(len(self.gradients))]
		#rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay_rate)
		adam = tf.train.AdamOptimizer(learning_rate=lr)
		self.update = adam.apply_gradients(zip(self.grads_buffer_ph, self.network_params))


	def create_network(self):
		input_layer = tflearn.input_data(shape=[None, self.inp_dim])
		dense = tflearn.fully_connected(input_layer, 10, activation='relu', name='hidden_layer')
		action_probs = tflearn.fully_connected(dense, self.out_dim, activation='softmax')
		#action_probs = tflearn.fully_connected(dense, 1, activation='softmax')

		return input_layer, action_probs

		"""observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
		W1 = tf.get_variable("W1", shape=[4, 10],
		           initializer=tf.contrib.layers.xavier_initializer())
		layer1 = tf.nn.relu(tf.matmul(observations,W1))
		W2 = tf.get_variable("W2", shape=[10, 1],
		           initializer=tf.contrib.layers.xavier_initializer())
		score = tf.matmul(layer1,W2)
		probability = tf.nn.sigmoid(score)"""

		return observations, probability


	def build_train_data(self):
		N = len(self.ys)
		y = np.eye(self.out_dim)[self.ys]
		return np.vstack(self.X), y, np.vstack(self.rs)

	def update_params(self):
		feed_dict = {}
		for i, grad in enumerate(self.grads_buffer_ph):
			feed_dict[self.grads_buffer_ph[i]] = self.grads_buffer[i]

		self.sess.run(self.update, feed_dict=feed_dict)

		for i,grad in enumerate(self.grads_buffer):
			self.grads_buffer[i] = grad*0

	def accumulate_gradients(self, **kwargs):
		inputs, y_labels, rewards = self.build_train_data()
		discounted_rewards = self.discount_rewards(rewards, **kwargs)
		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)

		gradients = self.sess.run(self.gradients, feed_dict={
				self.input_layer: inputs,
				self.y_fake_label: y_labels,
				self.reward_signal: discounted_rewards			
			})

		for i, grad in enumerate(gradients):
			self.grads_buffer[i] += grad

		#self.X, self.actions, self.rs = [],[],[]
		self.X, self.ys, self.rs = [],[],[]

	def predict(self, inputs):
		return self.sess.run(self.action_probs, feed_dict={
				self.input_layer: inputs
			})


	def record(self, x, action, reward):
		self.X.append(x)
		#y = 1 if action == 0 else 0 # a "fake label"
		#self.ys.append(y)
		self.ys.append(action)
		#self.actions.append(action)
		self.rs.append(reward)


	def discount_rewards(self, r, **kwargs):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		assert 'gamma' in kwargs, "Hey no gamma!"
		gamma = kwargs['gamma']
		for t in reversed(xrange(0, r.size)):
			if 'env_name' in kwargs:
				if kwargs['env_name'] == 'Pong-v0':
					if(r[t] != 0): runnnig_add = 0
			running_add = running_add * gamma + r[t]
			discounted_r[t] = running_add
		return discounted_r


	def save_model(self, ep_no):
		self.saver.save(self.sess, MODEL_DIR + "my_model", global_step = ep_no)


	def build_summaries(self): 
		episode_reward = tf.Variable(0.)
		tf.scalar_summary("Reward", episode_reward)
		summary_vars = [episode_reward]
		summary_ops = tf.merge_all_summaries()
		return summary_ops, summary_vars

	def initialize(self):
		self.summary_ops, self.summary_vars = self.build_summaries()
		self.sess.run(tf.initialize_all_variables())
		self.writer = tf.train.SummaryWriter(SUMMARY_DIR, self.sess.graph)
		self.saver = tf.train.Saver()
		self.X, self.actions, self.rs = [],[],[]

		self.grads_buffer = self.sess.run(self.network_params)
		for i,grad in enumerate(self.grads_buffer):
			self.grads_buffer[i] = grad*0

	def record_summary(self, episode_reward, episode_number):
		summary_str = self.sess.run(self.summary_ops, feed_dict={
				self.summary_vars[0]: episode_reward
			})

		self.writer.add_summary(summary_str, episode_number)
		self.writer.flush()
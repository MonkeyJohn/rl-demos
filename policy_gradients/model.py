import numpy as np
import tensorflow as tf
import sys, os

tf.logging.set_verbosity(tf.logging.ERROR)

class FCNet:
	def __init__(self, inp_dim, num_hidden, out_dim, lr=1e-2, decay_rate=0.99, log_dir="logs/", model_dir="saved_models/", model_name="model"):
		self.sess = tf.Session()
		self.inp_dim = inp_dim
		self.num_hidden = num_hidden
		self.out_dim = out_dim
		self.lr = lr
		self.decay_rate = decay_rate
		self.log_dir = log_dir
		self.model_dir = model_dir
		self.model_name = model_name + ".ckpt"
		
		self.X, self.ys, self.rs = [],[],[]

		self.input_layer, self.action_probs = self.create_network()
		self.network_params = tf.trainable_variables()
		
		self.y_labels = tf.placeholder(tf.float32, shape=[None, out_dim], 
										name='fake_label')
		self.advantage = tf.placeholder(tf.float32, name='reward_signal')
		self.logprobs = tf.log(tf.reduce_sum(tf.mul(self.y_labels, self.action_probs), 1))
		self.loss = -tf.reduce_mean(self.logprobs*tf.squeeze(self.advantage))
		self.gradients = tf.gradients(self.loss, self.network_params)

		
		self.grads_buffer_ph = [tf.placeholder(tf.float32) for i in range(len(self.gradients))]
		rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay_rate)
		#adam = tf.train.AdamOptimizer(learning_rate=lr)
		#self.update = adam.apply_gradients(zip(self.grads_buffer_ph, self.network_params))
		self.update = rmsprop.apply_gradients(zip(self.grads_buffer_ph, self.network_params))


	def create_network(self):
		import tflearn
		input_layer = tflearn.input_data(shape=[None, self.inp_dim], name='input_layer')
		dense = tflearn.fully_connected(input_layer, self.num_hidden, activation='relu', name='hidden_layer')
		action_probs = tflearn.fully_connected(dense, self.out_dim, activation='softmax', name='softmax_probs')

		return input_layer, action_probs


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

	def calculate_gradients(self, discounted_rewards):
		inputs, y_labels, rewards = self.build_train_data()
		gradients = self.sess.run(self.gradients, feed_dict={
				self.input_layer: inputs,
				self.y_labels: y_labels,
				self.advantage: discounted_rewards			
			})

		for i, grad in enumerate(gradients):
			self.grads_buffer[i] += grad

		self.X, self.ys, self.rs = [],[],[]

	def predict(self, inputs):
		return self.sess.run(self.action_probs, feed_dict={
				self.input_layer: inputs
			})


	def record(self, x, action, reward):
		self.X.append(x)
		self.ys.append(action)
		self.rs.append(reward)


	def save_model(self, ep_no=None):
		self.saver.save(self.sess, self.model_dir+self.model_name, global_step = ep_no)

	def load_model(self):
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			print "No saved model!"
			sys.exit(0)

	def build_summaries(self): 
		episode_reward = tf.Variable(0.)
		tf.scalar_summary("Reward_" + self.model_name, episode_reward)
		summary_vars = [episode_reward]
		summary_ops = tf.merge_all_summaries()
		return summary_ops, summary_vars

	def initialize(self):
		self.summary_ops, self.summary_vars = self.build_summaries()
		self.sess.run(tf.initialize_all_variables())
		self.writer = tf.train.SummaryWriter(self.log_dir, self.sess.graph)
		self.X, self.actions, self.rs = [],[],[]

		self.grads_buffer = self.sess.run(self.network_params)
		for i,grad in enumerate(self.grads_buffer):
			self.grads_buffer[i] = grad*0

		self.saver = tf.train.Saver()

	def record_summary(self, episode_reward, episode_number):
		summary_str = self.sess.run(self.summary_ops, feed_dict={
				self.summary_vars[0]: episode_reward
			})

		self.writer.add_summary(summary_str, episode_number)
		self.writer.flush()

"""logprobs = tf.nn.log_softmax(softmax_inputs)
action_probs = tf.exp(logprobs)
logp = tf.reduce_sum(logprobs * y_label, [1])  # y_label is at one-hot representation here
loss = - tf.reduce_mean(logp * advantage)"""



class FCNetDistributed:
	def __init__(self, inp_dim, num_hidden, out_dim, config, cluster, server, lr=1e-2, decay_rate=0.99, log_dir="logs/", model_dir="saved_models/", model_name="model"):
		self.inp_dim = inp_dim
		self.num_hidden = num_hidden
		self.out_dim = out_dim
		self.log_dir = log_dir
		self.model_dir = model_dir
		self.model_name = model_name + ".ckpt"
		
		self.config = config
		self.cluster = cluster
		self.server = server

		self.X, self.ys, self.rs = [],[],[]

		self.create_network()


	def create_network(self, lr=1e-2, decay_rate=0.99):
		is_chief = (self.config['task_id'] == 0)
		with tf.device(tf.train.replica_device_setter(cluster=self.cluster)):
			import tflearn as nn  # IMPORTANT TO BE HERE. FOR TFLEARN TO WORK

			self.input_layer = nn.input_data(shape=[None, self.inp_dim], 
										name='input_layer')
			dense = nn.fully_connected(self.input_layer, 200, 
										activation='relu', name='hidden_layer')
			self.action_probs = nn.fully_connected(dense, self.out_dim, 
										activation='softmax', 
										name='softmax_probs')

			self.network_params = tf.trainable_variables()
			self.y_labels = tf.placeholder(tf.float32, 
										shape=[None, self.out_dim],
										name='fake_label')
			self.advantage = tf.placeholder(tf.float32, name='reward_signal')

			logprobs = tf.log(tf.reduce_sum(tf.mul(self.y_labels, 
											self.action_probs), 1))
			self.loss = -tf.reduce_mean(logprobs*tf.squeeze(self.advantage))
			self.global_step = tf.Variable(0, name='global_step')

			self.gradients = tf.gradients(self.loss, self.network_params)
			self.grads_buffer_ph = [tf.placeholder(tf.float32) 
									for i in range(len(self.gradients))]
			rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr, 
												decay=decay_rate)
			self.update = rmsprop.apply_gradients(zip(self.grads_buffer_ph, 					self.network_params), 
									global_step=self.global_step)

			saver = tf.train.Saver()
			self.episode_reward = tf.Variable(0., name='episode_reward')
			self.summary_ops, self.summary_vars = self.build_summaries()
			init_op = tf.initialize_all_variables()

		self.sv = tf.train.Supervisor(is_chief=is_chief,
						 init_op=init_op,
						 summary_op=None,  # disable summary thread
						 saver=saver,
						 global_step=self.global_step,
						 save_model_secs=1800,
						 checkpoint_basename=self.model_name)

	def build_train_data(self):
		N = len(self.ys)
		y = np.eye(self.out_dim)[self.ys]
		return np.vstack(self.X), y

	def update_params(self):
		feed_dict = {}
		for i, grad in enumerate(self.grads_buffer_ph):
			feed_dict[self.grads_buffer_ph[i]] = self.grads_buffer[i]

		with self.sv.managed_session(self.server.target) as sess:
			sess.run(self.update, feed_dict=feed_dict)

		for i,grad in enumerate(self.grads_buffer):
			self.grads_buffer[i] = grad*0.0

	def calculate_gradients(self, discounted_rewards):
		with self.sv.managed_session(self.server.target) as sess:
			inputs, y_labels = self.build_train_data()
			gradients = sess.run(self.gradients, feed_dict={
									self.input_layer: inputs,
									self.y_labels: y_labels,
									self.advantage: discounted_rewards	
								})

		for i, grad in enumerate(gradients):
			self.grads_buffer[i] += grad

		self.X, self.ys, self.rs = [],[],[]

	def predict(self, inputs):
		with self.sv.managed_session(self.server.target) as sess:
			return sess.run(self.action_probs, feed_dict={
							self.input_layer: inputs
						})


	def record(self, x, action, reward):
		self.X.append(x)
		self.ys.append(action)
		self.rs.append(reward)

	def load_model(self):
		"TODO: For distribured version. Doesn't work now"
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			print "No saved model!"
			sys.exit(0)

	def build_summaries(self):
		tf.scalar_summary("Reward_" + self.model_name, self.episode_reward)
		summary_vars = [self.episode_reward]
		summary_ops = tf.merge_all_summaries()
		return summary_ops, summary_vars

	def initialize(self):
		with self.sv.managed_session(self.server.target) as sess:
			self.writer = tf.train.SummaryWriter(self.log_dir, sess.graph)
			self.grads_buffer = sess.run(self.network_params)

		self.X, self.actions, self.rs = [],[],[]
		for i,grad in enumerate(self.grads_buffer):
			self.grads_buffer[i] = grad*0

	def record_summary(self, episode_reward, episode_number):
		with self.sv.managed_session(self.server.target) as sess:
			summary_str = sess.run(self.summary_ops, feed_dict={
					self.summary_vars[0]: episode_reward
					})

		self.writer.add_summary(summary_str, episode_number)
		self.writer.flush()

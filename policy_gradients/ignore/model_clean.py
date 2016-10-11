import tensorflow as tf

def two_layer_network(inp_dim, num_hidden, out_dim, lr=1e-2, decay_rate=0.99, model_name='FCNet'):
	import tflearn as nn
	input_layer = nn.input_data(shape=[None,inp_dim], name='input_layer')

	dense = nn.fully_connected(input_layer, num_hidden, 
								activation='relu', name='hidden_layer',
								weights_init='xavier')
	action_probs = nn.fully_connected(dense, out_dim, 
								activation='softmax', 
								name='softmax_probs',
								weights_init='xavier')

	network_params = tf.trainable_variables()
	y_labels = tf.placeholder(tf.float32, 
								shape=[None, out_dim],
								name='fake_label')
	advantage = tf.placeholder(tf.float32, name='reward_signal')

	logprobs = tf.log(tf.reduce_sum(tf.mul(y_labels, action_probs), 1))
	loss = -tf.reduce_mean(logprobs*tf.squeeze(advantage))
	global_step = tf.Variable(0, name='global_step')

	gradients = tf.gradients(loss, network_params)
	grads_buffer_ph = [tf.placeholder(tf.float32) 
							for i in range(len(gradients))]

	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	update = optimizer.apply_gradients(zip(grads_buffer_ph, network_params), 
										global_step=global_step)

	saver = tf.train.Saver()
	episode_reward = tf.Variable(0., name='episode_reward')
	init_op = tf.initialize_all_variables()

	tf.scalar_summary("Reward_" + model_name, episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.merge_all_summaries()

	model = {}
	model['inp_dim'], model['out_dim'] = inp_dim, out_dim
	model['input_layer_ph'], model['y_labels_ph'] = input_layer, y_labels
	model['advantage_ph'], model['grads_buffer_ph'] = advantage, grads_buffer_ph
	model['network_params'] = network_params
	model['gradients_op'], model['predict_op'] = gradients, action_probs
	model['update_op'] = update

	model['init_op'],model['global_step'] = init_op, global_step
	model['saver'], model['episode_reward'] = saver, episode_reward
	model['summary'] = [summary_ops, summary_vars]

	return model
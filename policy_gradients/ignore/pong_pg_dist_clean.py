import tensorflow as tf
from util import create_cluster
import gym
from model_clean import two_layer_network
from pg_agent_clean import PGAgent
import os
import numpy as np

#flags for cluster
tf.app.flags.DEFINE_string("config_path", "./tmp/cfg",
							"path for cluster config files")
tf.app.flags.DEFINE_integer("node_id", 0, "Index of node in the cluster")
tf.app.flags.DEFINE_integer("n_ps", 1 , "Number of param servers in cluster")
tf.app.flags.DEFINE_integer("n_nodes", 3 , "Number of nodes in cluster")
tf.app.flags.DEFINE_integer("timeout", 0,
							"Time (s) to wait for cluster to come online")
tf.app.flags.DEFINE_integer("aws", 0, "Running on AWS or not")

#learning flags
tf.app.flags.DEFINE_integer("n_steps", 100 , "Number of nodes in cluster")
tf.app.flags.DEFINE_integer("batch_size", 128 , "Batch size")

flags = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.ERROR)

DEFAULT_PORT = 5555
def cluster_config(flags):
	n_nodes, node_id, n_ps = flags.n_nodes, flags.node_id, flags.n_ps
	config = {}
	config['ps_hosts'] = ['master:%d' % DEFAULT_PORT]
	config['worker_hosts'] = []
	host_base = 'node0'
	for i in range(1, n_nodes):
		if i < n_ps:
			n_str = str(i) if i >= 10 else '0' + str(i)
			config['ps_hosts'].append(host_base + n_str + ':' + str(DEFAULT_PORT + i))
		else:
			n_str = str(i) if i >= 10 else '0' + str(i)
			config['worker_hosts'].append(host_base + n_str +  ':' + str(DEFAULT_PORT + i))

	if node_id < n_ps:
		config['job'] = 'ps'
		config['task_id'] = node_id
	else:
		config['job'] = 'worker'
		config['task_id'] = node_id - n_ps

	return config

def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel().reshape((1,6400))

def map_action(action_space, action, env_to_agent=True):
	if env_to_agent:
		d = {2: 0, 3: 1}
		return d[action]
	else:
		d = {0: 2, 1: 3}
		return d[action]


def main(args):
	if not os.path.isdir(flags.config_path):
		os.makedirs(flags.config_path)

	if flags.aws == 0:
		config = create_cluster(flags.node_id, flags.config_path, flags.n_nodes,
							flags.n_ps, flags.timeout)
	else:
		config = cluster_config(flags)

	# Create a cluster from the parameter server and worker hosts.
	cluster = tf.train.ClusterSpec({"ps": config['ps_hosts'],
									"worker": config['worker_hosts']})

	print config['ps_hosts'], config['worker_hosts'], config['job'], config['task_id']
	# Create and start a server for the local task.
	server = tf.train.Server(cluster,
							 job_name=config['job'],
							 task_index=config['task_id'])

	if config['job'] == 'ps':
		server.join()
	elif config['job'] == 'worker':
		is_chief = (config['task_id'] == 0)

		with tf.device(tf.train.replica_device_setter(cluster=cluster)):
			env = gym.make('Pong-v0')
			state_dim = env.observation_space.shape[0]
			action_dim = env.action_space.n

			model = two_layer_network(6400, 200, 2)

		init_op, saver, global_step = model['init_op'], model['saver'], model['global_step']
		sv = tf.train.Supervisor(is_chief=is_chief,
						 logdir="./tmp/train_logs",
						 init_op=init_op,
						 summary_op=None,  # disable summary thread;crashes
						 saver=saver,
						 global_step=global_step,
						 save_model_secs=600)

		with sv.managed_session(server.target) as sess:
			agent = PGAgent(env, 'Pong-v0', sess, model, pre_processor=prepro, action_space=[2,3], map_action=map_action)
			agent.learn(render=False, diff_frame=True, batch_size=20, max_episodes=20000)

if __name__ == "__main__":
	tf.app.run()
import tensorflow as tf
from util import create_cluster
import gym
from model_clean import two_layer_network
from pg_agent_clean import PGAgent
import os

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
			env = gym.make('CartPole-v0')
			state_dim = env.observation_space.shape[0]
			action_dim = env.action_space.n

			model = two_layer_network(state_dim, 200, action_dim)

		init_op, saver, global_step = model['init_op'], model['saver'], model['global_step']
		sv = tf.train.Supervisor(is_chief=is_chief,
						 logdir="./tmp/train_logs",
						 init_op=init_op,
						 summary_op=None,  # disable summary thread;crashes
						 saver=saver,
						 global_step=global_step,
						 save_model_secs=600)

		with sv.managed_session(server.target) as sess:
			agent = PGAgent(env, 'CartPole-v0', sess, model)
			agent.learn(render=False, diff_frame=False, batch_size=25, max_episodes=4000)

if __name__ == "__main__":
	tf.app.run()
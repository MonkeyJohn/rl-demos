import tensorflow as tf
import gym
from model import FCNetDistributed
from pg_agent import PGAgent
import os

#flags for cluster
tf.app.flags.DEFINE_string("config_path", "./tmp/cfg",
							"path for cluster config files")
tf.app.flags.DEFINE_integer("node_id", 0, "Index of node in the cluster")
tf.app.flags.DEFINE_integer("n_ps", 1 , "Number of param servers in cluster")
tf.app.flags.DEFINE_integer("n_nodes", 3 , "Number of nodes in cluster")
tf.app.flags.DEFINE_integer("timeout", 0,
							"Time (s) to wait for cluster to come online")

#learning flags
tf.app.flags.DEFINE_integer("n_steps", 100 , "Number of nodes in cluster")
tf.app.flags.DEFINE_integer("batch_size", 128 , "Batch size")

flags = tf.app.flags.FLAGS


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

def main(args):
	if not os.path.isdir(flags.config_path):
		os.makedirs(flags.config_path)

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
		env = gym.make('CartPole-v0')
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.n
		print "Before FCNet"
		model = FCNetDistributed(state_dim, 10, action_dim, config, cluster, 
								server, model_name='FCNetDistributed')
		print "Before PGAgent"
		agent = PGAgent(env,'CartPole-v0', model)
		print "Entering Learn"
		agent.learn(render=False, diff_frame=False, batch_size=50, 
					max_episodes=4000)


if __name__ == "__main__":
	tf.app.run()
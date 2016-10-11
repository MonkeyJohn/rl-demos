def cluster_config(flags):
	DEFAULT_PORT = 6777
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
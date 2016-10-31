import numpy as np

def cluster_config(flags):
	DEFAULT_PORT = 7777
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


def preprocess(I, bg_colors=[144, 109]):
    """ Atari frame preprocessing (Karpathy)

    Reduces atari color frame to 80x80 B&W image
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector

    Args:
        I (ndarray): 210x160x3 uint8 frame
        bg_colors (iterable): list of uint8 colors to zero out

    Returns:
         numpy array (1,6400) with dtype float representing 80x80 B&W image
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    for c in bg_colors:
        I[I == c] = 0  # erase background
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).reshape((1, -1))

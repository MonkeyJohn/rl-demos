import time
import os
import json
import socket
import glob
import tensorflow as tf

DEFAULT_PORT = 22600


def write_config(node_id, path='.', n_nodes=10, n_ps=2):
    '''
    Determine local cluster node config and write json config file.
    '''
    cfg = {}
    cfg['host'] = socket.gethostname()
    cfg['port'] = DEFAULT_PORT + node_id
    if node_id < n_ps:
        cfg['job'] = 'ps'
        cfg['task_id'] = node_id
    else:
        cfg['job'] = 'worker'
        cfg['task_id'] = node_id - n_ps
    name = os.path.join(path, 'node%d.json' % node_id)
    with open(name, 'w') as f:
        json.dump(cfg, f)
    return cfg


def get_cluster_spec(path='.'):
    '''
    Read cluster spec from directory containing node config files
    '''
    cfg_files = [f for f in glob.glob(os.path.join(path, 'node*.json'))]
    # make sure hosts are ordered by task_id
    sorted(cfg_files)
    worker_hosts = []
    ps_hosts = []
    for cfg in cfg_files:
        with open(cfg, 'r') as f:
            node_cfg = json.load(f)
        host_address = str(node_cfg['host'])+':'+str(node_cfg['port'])
        if node_cfg['job'] == 'ps':
            ps_hosts.append(host_address)
        elif node_cfg['job'] == 'worker':
            worker_hosts.append(host_address)
        else:
            raise RuntimeError('Illegal config file')
    return ps_hosts, worker_hosts


def create_cluster(node_id, path='.', n_nodes=10, n_ps=2, timeout=0):
    ''' Setup method for tensorflow cluster

    Determines role of local node, writes config to shared directory and then
    waits for all other nodes to come online and do the same.

    Args:
        node_id (int): id of the local node. Must be unique in cluster
        path (str): path of configuration directory, where nodes will store
            their config files. Needs to be on a shared file system readable
            and writable by all nodes.
        n_nodes (int): total number of nodes in the cluster
        n_ps (int): number of parameter servers in cluster. All other nodes
            will be worker nodes.
        timeout (int): maximum number of seconds to wait for nodes to come
            online. 0 (default) waits forever.
    Returns:
        dictionary containing all cluster settings needed for tensorflow
        cluster spec and tensorflow server.
    '''
    # get config for this node and write to config folder
    cfg = write_config(node_id, path, n_nodes, n_ps)
    # wait for other nodes to come up
    node_count = len(glob.glob(os.path.join(path, 'node*.json')))
    start_time = time.time()
    while node_count < n_nodes:
        if 0 < timeout < time.time()-start_time:
            raise RuntimeError('Cluster startup time out')
        time.sleep(2)
        node_count = len(glob.glob(os.path.join(path, 'node*.json')))
    # read list of hosts and ports and return spec for local server
    ps_hosts, worker_host = get_cluster_spec(path)
    cfg['ps_hosts'] = ps_hosts
    cfg['worker_hosts'] = worker_host
    return cfg


def run_distributed_graph(create_graph, generate_feed, flags):
    ''' Run distributed graph optimization

    Function to optimize a graph using replication with asynchronous updating.
    Computational graph and data generation can be passed as arguments.
    Args:
    create_graph: function that builds computational graph. Takes as input a tf
        flags object. Should return the target variable that needs to be
        optimized, as well as an info object  (dict) that contains additional
        information needed for the optimization procedure. The target variable
        will be passed to the optimizers minimize method. The info dict will
        be passed (unaltered) to the data generator.
    generate_feed: function that creates the data generator. Takes info object
        (see above) and tf flags objects as input and returns a generator
        object. Every call to the next() method of this generator should return
        a feed dict containing a new batch of data for the next step of the
        optimizer.
    flags: tensorflow flags object. Should contain all configuration flags
        expected by the methods above. Additionally it should specify following
        settings:
        - node_id (int) in range(0,num_nodes)
        - confif_path (str) shared directory for node config files
        - n_nodes (int) total number of nodes in the cluster
        - n_ps (int) number of parameter servers in cluster
        - timeout (int) seconds to wait for cluster to come online

    This function should be run on every node in the cluster.
    '''
    if not os.path.isdir(flags.config_path):
        os.makedirs(flags.config_path)

    config = create_cluster(flags.node_id, flags.config_path, flags.n_nodes,
                            flags.n_ps, flags.timeout)
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": config['ps_hosts'],
                                    "worker": config['worker_hosts']})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=config['job'],
                             task_index=config['task_id'])

    if config['job'] == "ps":
        server.join()
    elif config['job'] == "worker":
        is_chief = (config['task_id'] == 0)

        # Assigns ops to the local worker by default.
        # note: this automatically sets the device on which ops/vars are stored
        # ops are stored on the local worker running this code, vars are stored
        # on param server - so each worker has network copy but actual weight
        # values are shared through paramserver
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            # worker_device="/job:worker/task:%d" % FLAGS.task_index,
            info, target = create_graph(flags)
            global_step = tf.Variable(0)

            # split minimize into compute and apply gradients for optional
            # gradient processing (logging, clipping,...)
            optimizer = tf.train.AdagradOptimizer(0.01)
            grads_and_vars = optimizer.compute_gradients(target)
            optimizer = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step)

            # create ops for saving, logging and initializing
            saver = tf.train.Saver()
            # summary_op = tf.merge_all_summaries() if is_chief else None
            init_op = tf.initialize_all_variables()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir="./tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=None,  # disable summary thread
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        print 'Hi'

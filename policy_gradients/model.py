import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

def two_layer_net(inp_dim, num_hidden, out_dim):
    S = tf.placeholder(shape=(None, inp_dim), dtype=tf.float32, name='obs')
    A = tf.placeholder(shape=(None,), dtype=tf.int32, name='acts')
    Adv = tf.placeholder(shape=(None,), dtype=tf.float32, name='advs')

    # Using TFLearn wrappers for network building
    import tflearn as nn  # needs to be here since tflearn defines vars
    # simple mlp with 1 hidden layer
    net = nn.fully_connected(S, num_hidden, activation='relu')
    #net = nn.dropout(net, 0.8)
    # could use sigmoid for only 2 actions, but this is more general
    net = nn.fully_connected(net, out_dim, activation='softmax')
    network_params = tf.trainable_variables()
    # probabilities of selected actions
    # note: need to reduce to single dimension because of tf
    # indexing limitations
    idx = tf.range(tf.shape(net)[0])
    probs = tf.gather(tf.reshape(net, [-1]), idx * tf.shape(net)[1] + A)
    # Defining pg loss using Tensorflow
    # loss = -tf.matmul(tf.log(probs), Adv)/tf.shape(Adv)[0]
    loss = -tf.reduce_mean(tf.log(probs) * Adv)
    global_step = tf.Variable(0)

    # split minimize into compute and apply gradients for optional gradient
    # processing (logging, clipping,...)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99)
    gradients = tf.gradients(loss, network_params)
    grads_buffer_ph = [tf.placeholder(tf.float32) for i in range(len(gradients))]

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    optimizer = optimizer.apply_gradients(zip(grads_buffer_ph, network_params),
                                          global_step=global_step)

    # create ops for saving, logging and initializing
    saver = tf.train.Saver()
    # summary_op = tf.merge_all_summaries() if is_chief else None
    init_op = tf.initialize_all_variables()

    model = {}
    model['inp_dim'], model['num_hidden'], model['out_dim'] = inp_dim, num_hidden, out_dim
    model['input_ph'], model['actions_ph'], model['advantage_ph'] = S, A, Adv
    model['gradients'], model['grads_buffer_ph'] = gradients, grads_buffer_ph
    model['network_params'] = network_params
    model['net'], model['loss'], model['optimizer'] = net, loss, optimizer
    model['saver'], model['init_op'], model['global_step'] = saver, init_op, global_step

    return model
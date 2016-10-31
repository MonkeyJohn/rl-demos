import tensorflow as tf
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from keras import backend as K


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


def two_layer_net_a3c(inp_dim, out_dim, num_hidden=256, lr=1e-4, decay=0.99):
    states = tf.placeholder(shape=(None, inp_dim), dtype=tf.float32, name='states')
    actions = tf.placeholder(shape=(None, out_dim), dtype=tf.float32, name='actions')
    rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')

    import tflearn as nn
    net = nn.fully_connected(states, num_hidden, activation='relu')

    probs = nn.fully_connected(net, out_dim, activation='softmax', name='policy')
    value = nn.fully_connected(net, 1, activation='linear', name='value')

    logprobs = tf.log(tf.reduce_sum(tf.mul(actions, probs), 1))

    advantage = rewards - value
    policy_loss = -tf.reduce_sum(logprobs*advantage, name='policy_loss')
    value_loss = tf.nn.l2_loss(advantage, name='value_loss')
    entropy_loss = -tf.reduce_sum(probs*tf.log(probs), name='entropy_loss')

    loss = policy_loss + value_loss - 0.01*entropy_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimize = optimizer.minimize(loss)

    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    running_reward = tf.placeholder(tf.float32, name='running_reward')
    episode_reward = tf.placeholder(tf.float32, name='episode_reward')
    tf.scalar_summary("Running Reward", running_reward)
    tf.scalar_summary("Episode Reward", episode_reward)
    summary_op = tf.merge_all_summaries()

    model = {}
    model['inp_dim'], model['out_dim'] = inp_dim, out_dim
    model['states'], model['actions'], model['rewards'] = states,actions,rewards
    model['probs'], model['value'] = probs, value
    model['optimize'] = optimize
    model['saver'], model['init_op'], model['summary_op'] = saver,init_op,summary_op
    model['running_reward'], model['episode_reward'] = running_reward, episode_reward

    return model 


def convnet_a3c(height, width, channels, num_actions=2, lr=1e-4, decay=0.99):
    from keras.layers import Convolution2D, Flatten, Dense, Input
    from keras.models import Model

    states = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
    actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name='actions')
    advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantage')
    rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')

    inputs = Input(shape=(height,width,channels))
    shared = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), 
                        border_mode='same', activation='relu', name='conv1')(inputs)
    shared = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), 
                        border_mode='same', activation='relu', name='conv2')(shared)
    shared = Flatten()(shared)
    shared = Dense(output_dim=256, activation='relu', name='fc1')(shared)

    action_probs = Dense(output_dim=num_actions, activation='softmax', 
                        name='probs')(shared)
    state_value  = Dense(output_dim=1, activation='linear', name='value')(shared)

    policy_network = Model(input=inputs, output=action_probs)
    value_network = Model(input=inputs, output=state_value)

    policy_params = policy_network.trainable_weights
    value_params = value_network.trainable_weights
    params = policy_params + value_params[-2:]

    probs, value = policy_network(states), value_network(states)
    log_probs = tf.log(tf.clip_by_value(probs, 1e-20, 1.0))
    log_probs_act = tf.reduce_sum(tf.mul(log_probs, actions), 1)

    policy_loss = -tf.reduce_sum(logprobs_act*advantage, name='policy_loss')
    value_loss = tf.nn.l2_loss(rewards - value, name='value_loss')
    entropy_loss = -tf.reduce_sum(probs*log_probs, name='entropy_loss')

    loss = policy_loss + 0.5*value_loss - 0.01*entropy_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimize = optimizer.minimize(loss)

    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    running_reward = tf.placeholder(tf.float32, name='running_reward')
    episode_reward = tf.placeholder(tf.float32, name='episode_reward')
    tf.scalar_summary("Running Reward", running_reward)
    tf.scalar_summary("Episode Reward", episode_reward)
    summary_op = tf.merge_all_summaries()

    model = {}
    model['states'], model['actions'], model['rewards'] = states,actions,rewards
    model['probs'], model['value'] = probs, value
    model['optimize'] = optimize
    model['saver'], model['init_op'], model['summary_op'] = saver,init_op,summary_op
    model['running_reward'], model['episode_reward'] = running_reward, episode_reward
    model['params'] = params
    model['value_params'], model['policy_params'] = value_params, policy_params
    return model


class ConvNetA3C:
    def __init__(self, height, width, channels, num_actions):
        self.create_network(height, width, channels, num_actions)
        self.build_summaries()

    def create_network(self, height, width, channels, num_actions):
        self.states = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
        self.actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name='actions')
        self.rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantage')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')

        inputs = Input(shape=(height,width,channels))
        shared = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), 
                            border_mode='valid', activation='relu', name='conv1',
                            dim_ordering='tf')(inputs)
        shared = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), 
                            border_mode='valid', activation='relu', name='conv2',
                            dim_ordering='tf')(shared)
        shared = Flatten()(shared)
        shared = Dense(output_dim=256, activation='relu', 
                            name='fc1')(shared)

        action_probs = Dense(output_dim=num_actions, activation='softmax',  
                             name='probs')(shared)
        state_value  = Dense(output_dim=1, activation='linear', 
                            name='value')(shared)

        self.policy_and_value_network = Model(input=inputs, output=[action_probs, state_value])
        self.params = self.policy_and_value_network.trainable_weights

        self.probs, self.value = self.policy_and_value_network(self.states)
        self.value = tf.reshape(self.value,[-1])

        log_probs = tf.log(tf.clip_by_value(self.probs, 1e-20, 1.0))
        log_probs_act = tf.reduce_sum(tf.mul(log_probs, self.actions), 1)

        entropy = -tf.reduce_sum(self.probs*log_probs, reduction_indices=1)
        policy_loss = -tf.reduce_sum(log_probs_act*self.advantage + 0.01*entropy, name='policy_loss')
        value_loss = tf.nn.l2_loss(self.rewards - self.value, name='value_loss')

        self.loss = policy_loss + 0.5*value_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimize = optimizer.minimize(self.loss)

    def sync_from(self, src_network):
        src_params = src_network.params
        dst_params = self.params

        sync_ops = []
        with tf.op_scope([], None, "ConvNetA3C") as name:
            for s, d in zip(src_params, dst_params):
                sync_op = tf.assign(d, s)
                sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def update(self, sess, states, actions, rewards, adv, lr):
        sess.run(self.optimize, feed_dict={
                                self.states: states,
                                self.actions: actions,
                                self.rewards: rewards,
                                self.advantage: adv,
                                self.lr: lr
            })

    def get_policy_and_value(self, sess, state):
        probs, value = sess.run([self.probs, self.value], feed_dict={
                            self.states: [state],
            })
        return (probs[0], value[0])

    def get_policy(self, sess, state):
        return sess.run(self.probs, feed_dict={
                            self.states: [state]
            })[0]

    def get_value(self, sess, state):
        return sess.run(self.value, feed_dict={
                            self.states: [state]
            })[0]


    def build_summaries(self):
        self.running_reward = tf.placeholder(tf.float32, name='running_reward')
        self.episode_reward = tf.placeholder(tf.float32, name='episode_reward')
        tf.scalar_summary("Running Reward", self.running_reward)
        tf.scalar_summary("Episode Reward", self.episode_reward)
        self.summary_op = tf.merge_all_summaries()

    def record_summary(self, sess, writer, avg_rwd, ep_rwd, T):
        summary = sess.run(self.summary_op, feed_dict={
                                self.running_reward: avg_rwd,
                                self.episode_reward: ep_rwd
            })

        writer.add_summary(summary, T)


class TwoLayerNetA3C:
    def __init__(self, state_dim, num_actions):
        self.create_network(state_dim, num_actions)
        self.build_summaries()

    def create_network(self, state_dim, num_actions):
        self.states = tf.placeholder(shape=(None, state_dim), dtype=tf.float32)
        self.actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name='actions')
        self.rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantage')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')

        inputs = Input(shape=(state_dim,))
        shared = Dense(output_dim=256, activation='relu', name='fc1')(inputs)

        action_probs = Dense(output_dim=num_actions, activation='softmax',  
                             name='probs')(shared)
        state_value  = Dense(output_dim=1, activation='linear', 
                            name='value')(shared)

        self.policy_and_value_network = Model(input=inputs, output=[action_probs, state_value])
        self.params = self.policy_and_value_network.trainable_weights

        self.probs, self.value = self.policy_and_value_network(self.states)
        self.value = tf.reshape(self.value,[-1])

        log_probs = tf.log(tf.clip_by_value(self.probs, 1e-20, 1.0))
        log_probs_act = tf.reduce_sum(tf.mul(log_probs, self.actions), 1)

        entropy = -tf.reduce_sum(self.probs*log_probs, reduction_indices=1)
        policy_loss = -tf.reduce_sum(log_probs_act*self.advantage + 0.01*entropy, name='policy_loss')
        value_loss = tf.nn.l2_loss(self.rewards - self.value, name='value_loss')

        self.loss = policy_loss + 0.5*value_loss
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.optimize = optimizer.minimize(self.loss)

    def sync_from(self, src_network):
        src_params = src_network.params
        dst_params = self.params

        sync_ops = []
        with tf.op_scope([], None, "TwoLayerNetA3C") as name:
            for s, d in zip(src_params, dst_params):
                sync_op = tf.assign(d, s)
                sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def update(self, sess, states, actions, rewards, adv, lr):
        sess.run(self.optimize, feed_dict={
                                self.states: states,
                                self.actions: actions,
                                self.rewards: rewards,
                                self.advantage: adv,
                                self.lr: lr
            })

    def get_policy_and_value(self, sess, state):
        probs, value = sess.run([self.probs, self.value], feed_dict={
                            self.states: [state],
            })
        return (probs[0], value[0])

    def get_policy(self, sess, state):
        return sess.run(self.probs, feed_dict={
                            self.states: [state]
            })[0]

    def get_value(self, sess, state):
        return sess.run(self.value, feed_dict={
                            self.states: [state]
            })[0]


    def build_summaries(self):
        self.running_reward = tf.placeholder(tf.float32, name='running_reward')
        self.episode_reward = tf.placeholder(tf.float32, name='episode_reward')
        tf.scalar_summary("Running Reward", self.running_reward)
        tf.scalar_summary("Episode Reward", self.episode_reward)
        self.summary_op = tf.merge_all_summaries()

    def record_summary(self, sess, writer, avg_rwd, ep_rwd, T):
        summary = sess.run(self.summary_op, feed_dict={
                                self.running_reward: avg_rwd,
                                self.episode_reward: ep_rwd
            })

        writer.add_summary(summary, T)
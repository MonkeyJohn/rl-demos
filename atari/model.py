import tensorflow as tf
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from keras import backend as K
# from rmsprop_applier import RMSPropApplier
# from accum_trainer import AccumTrainer


#tf.logging.set_verbosity(tf.logging.ERROR)

class ConvNetA3C:
    def __init__(self, height, width, channels, num_actions, summary=True):
        self.create_network(height, width, channels, num_actions)
        if summary:
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
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.optimize = optimizer.minimize(self.loss)

        # avoid NaN with clipping when value in pi becomes zero
        logprobs = tf.log(tf.clip_by_value(self.probs, 1e-20, 1.0))
        logprobs_act = tf.reduce_sum(tf.mul(logprobs, self.actions), 1)

        entropy = -tf.reduce_sum(self.probs * logprobs, reduction_indices=1)
        policy_loss = -tf.reduce_sum( logprobs_act*self.advantage + entropy*0.01)
        value_loss = 0.5 * tf.nn.l2_loss(self.rewards - self.value)

        self.loss = policy_loss + value_loss


    def sync_from(self, src_network):
        src_params = src_network.params
        dst_params = self.params

        sync_ops = []
        with tf.op_scope([], None, "ConvNetA3C") as name:
            for s, d in zip(src_params, dst_params):
                sync_op = tf.assign(d, s)
                sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    # def update(self, sess, states, actions, rewards, adv, lr):
    #     # sess.run(self.optimize, feed_dict={
    #     #                         self.states: states,
    #     #                         self.actions: actions,
    #     #                         self.rewards: rewards,
    #     #                         self.advantage: adv,
    #     #                         self.lr: lr
    #     #     })
    #     # sess.run(self.accum_gradients, feed_dict={
    #     #                         self.states: states,
    #     #                         self.actions: actions,
    #     #                         self.rewards: rewards,
    #     #                         self.advantage: adv,
    #     #     })
    #     # sess.run(self.apply_gradients,
    #     #       feed_dict = {self.lr: lr} )

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
        optimizer = tf.train.RMSPropOptimizer(self.lr)
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
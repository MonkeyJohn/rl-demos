import tensorflow as tf
import numpy as np
import time
from model import two_layer_net

'''
General utilities for policy gradient methods
'''



def discount_rewards(r, gamma, env_name, initial=None):
    """ take 1D float array of rewards and compute discounted reward
    Args:
        r:  1D array of episode reward
        gamma (float): discount facter

    Returns:
        1D numpy array (float):  discounted returns for each step

    """
    discounted_r = np.zeros_like(r)
    running_add = 0 if initial is None else initial
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def run_episode(env, model, server, checkpoint_dir, actions, preprocess=None):
    ''' Generate trajectory by running single episode

    Runs a single episode in gym environment using a neural network policy.
    Returns observation, action, reward trajectories

    Returns:

        tuple: ndarrays containing observations, actions and rewards
            respectively
    '''

    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # graph = tf.Graph()
    # sess = tf.Session(graph=graph)

    # with sess.graph.as_default():
    #     model = two_layer_net(model['inp_dim'], model['num_hidden'], model['out_dim'])
    #     net, saver = model['net'], model['saver']
    #     input_ph = model['input_ph']
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         # hack code
    #         print("NO CHECKPOINT FOUND!!!!!!!!!!!")
    #         print("waiting for sometime and trying again.")
    #         time.sleep(10)
    #         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #         if ckpt and ckpt.model_checkpoint_path:
    #             saver.restore(sess, ckpt.model_checkpoint_path)



    observation = env.reset()
    xs, acts, rs = [], [], []
    done = False
    while not done:
        if preprocess:
            cur_x = preprocess(observation)
        else:
            cur_x = observation.reshape((1, observation.shape[0]))

        x = cur_x

        feed_dict = {input_ph: x}

        aprob = sess.run(net, feed_dict=feed_dict)
        # sample action using probs
        act = np.random.choice(actions.size, p=aprob.flatten())

        # record trajectory
        xs.append(x)  # observation
        acts.append(act)  # action (index)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(actions[act])
        # record reward
        # note: has to be done after we call step()
        # to get reward for previous action
        rs.append(reward)

    return [np.vstack(xs), np.array(acts), np.array(rs)]


def run_episode_a3c(env, model, sess, actions, s_t=None,terminal=False, preprocess=None):
    t_max = 32

    state_ph, probs = model['states'], model['probs']
    
    xs, acts, rs = [], [], []
    t = 0
    observation = env.reset() if s_t is None else s_t

    while not (terminal or (t==t_max)):
        if preprocess:
            cur_x = preprocess(observation)
        else:
            cur_x = observation.reshape((1, observation.shape[0]))

        x = cur_x

        aprob = sess.run(probs, feed_dict={state_ph: x})[0]
        act = np.random.choice(actions.size, p=aprob.flatten())
        observation, r_t, terminal, info = env.step(actions[act])

        xs.append(x)
        acts.append(act)
        #r_t = np.clip(r_t, -1, 1)
        rs.append(r_t)

        t += 1

    return (np.vstack(xs), np.array(acts), np.array(rs), observation, terminal)
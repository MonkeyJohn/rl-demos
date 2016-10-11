import tensorflow as tf
import numpy as np
import time
from model import two_layer_net

'''
General utilities for policy gradient methods
'''


def prepro(I, bg_colors=[144, 109]):
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



def discount_rewards(r, gamma, env_name):
    """ take 1D float array of rewards and compute discounted reward
    Args:
        r:  1D array of episode reward
        gamma (float): discount facter

    Returns:
        1D numpy array (float):  discounted returns for each step

    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if env_name == 'Pong-v0':
            if r[t] != 0:               
                running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def run_episode(env, model, server, checkpoint_dir, actions, diff_frame=True, preprocess=True):
    ''' Generate trajectory by running single episode

    Runs a single episode in gym environment using a neural network policy.
    Returns observation, action, reward trajectories

    Returns:

        tuple: ndarrays containing observations, actions and rewards
            respectively
    '''

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with sess.graph.as_default():
        model = two_layer_net(model['inp_dim'], model['num_hidden'], model['out_dim'])
        net, saver = model['net'], model['saver']
        input_ph = model['input_ph']
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # hack code
            print("NO CHECKPOINT FOUND!!!!!!!!!!!")
            print("waiting for sometime and trying again.")
            time.sleep(10)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)



    observation = env.reset()
    prev_x = None
    xs, acts, rs = [], [], []
    done = False
    while not done:
        # TODO: implement deepmind preprocessing with histories
        # preprocess the current observation
        if preprocess:
            cur_x = prepro(observation)
        else:
            cur_x = observation.reshape((1, observation.shape[0]))
        # set input to network to be difference image
        if diff_frame:
            x = cur_x - prev_x if prev_x is not None else np.zeros((1, cur_x.size))
        else:
            x = cur_x
            
        prev_x = cur_x
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

#        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
#            print ('ep %d: game finished, reward: %f' % (n_e, reward)) + ('' if reward == -1 else ' !!!!!!!!')
    return [np.vstack(xs), np.array(acts), np.array(rs)]

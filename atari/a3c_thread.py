import tensorflow as tf
from util_pg import run_episode_a3c, discount_rewards
from util import preprocess
import gym
import numpy as np
from model import two_layer_net_a3c
import time
import threading

tf.app.flags.DEFINE_string("env", "Pong-v0", "Gym environment to run")
tf.app.flags.DEFINE_float("gamma", .99, "Discount factor")
tf.app.flags.DEFINE_integer("num_episodes", 20000,
                            "Total no. of episodes to run")
tf.app.flags.DEFINE_integer("num_steps", 1000000, "Maximum number of updates")
tf.app.flags.DEFINE_boolean("preprocess", True, "Whether to preprocess input")
tf.app.flags.DEFINE_string("actions", "2,3", "action space as comma separated strings")
tf.app.flags.DEFINE_integer("inp_dim", 80*80, "Input dimension for problem")
tf.app.flags.DEFINE_integer("out_dim", 2, "Output dimension for problem")
tf.app.flags.DEFINE_boolean("train", True, "Training phase or Eval phase")

FLAGS = tf.app.flags.FLAGS

NUM_THREADS = 2
LOGDIR_TRAIN = FLAGS.env + "a3c_train_logs/"
LOGDIR_EVAL  = FLAGS.env + "a3c_test_logs/"
SAVE_EVERY = 2500
T = 0

def a3c_thread(t_id, env, sess, model, actions, writer):
    global T

    state_ph, actions_ph = model['states'],model['actions']
    rewards_ph = model['rewards']
    probs, value = model['probs'], model['value']
    optimize = model['optimize']
    init_op, summary_op = model['init_op'], model['summary_op']
    running_reward_ph = model['running_reward']
    episode_reward_ph = model['episode_reward']
    saver = model['saver']
    check_process, gamma, env_name = FLAGS.preprocess, FLAGS.gamma, FLAGS.env
    pre = preprocess if check_process else None

    avg_reward  = None
    n_e = 0
    terminal = False
    s_t = None
    ep_rwd = 0

    while n_e < FLAGS.num_episodes:
        o_n, a_n, r_n, s_t, prev_x, terminal = run_episode_a3c(env, 
                                                       model,
                                                       sess, 
                                                       actions, 
                                                       s_t, 
                                                       terminal,
                                                       preprocess=pre)
        ep_rwd += np.sum(r_n)

        if terminal:
            R = 0
        else:
            x = pre(s_t) - prev_x
            R = sess.run(value, feed_dict={state_ph:x})[0][0]

        disc_r = discount_rewards(r_n, gamma, env_name, initial=R)

        sess.run(optimize, feed_dict={
                            state_ph: o_n,
                            actions_ph: np.eye(actions.size)[a_n],
                            rewards_ph: disc_r })

        if terminal:
            avg_reward = ep_rwd if avg_reward is None\
                                else 0.99*avg_reward + 0.01*ep_rwd

            print "Thread %d, Episode %d, Avg Reward %1.3f and Episode Reward %1.3f"\
                                                    % (t_id, n_e, avg_reward, ep_rwd)
            
            summary = sess.run(summary_op, feed_dict={
                                            running_reward_ph: avg_reward,
                                            episode_reward_ph: ep_rwd})
            ep_rwd = 0
            terminal = False
            s_t = None # reset
            n_e += 1
            writer.add_summary(summary, n_e)

        if T % SAVE_EVERY == 0:
            saver.save(sess, LOGDIR_TRAIN, global_step=T)

        T += 1

def train(sess, model):
    writer = tf.train.SummaryWriter(LOGDIR_TRAIN, graph=sess.graph)
    actions = np.array(map(int, FLAGS.actions.split(',')))
    envs = [gym.make(FLAGS.env) for i in range(NUM_THREADS)]

    threads = [threading.Thread(target=a3c_thread, args=(i, envs[i],
                                                                sess, 
                                                                model,
                                                                actions,
                                                                writer))
                                                for i in range(NUM_THREADS)]
    sess.run(tf.initialize_all_variables())
    for th in threads:
        th.start()

    for th in threads:
        th.join()
        print "A3c thread finished"



def evaluation(sess, model):
    pass

def main(args):
    with tf.Session() as sess:
        model = two_layer_net_a3c(FLAGS.inp_dim, FLAGS.out_dim)
        if FLAGS.train:
            train(sess, model)
        else:
            evaluation(sess, model)


if __name__ == "__main__":
    tf.app.run()
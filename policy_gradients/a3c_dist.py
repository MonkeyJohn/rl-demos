import tensorflow as tf
from util_pg import run_episode_a3c, discount_rewards
from util import preprocess
import gym
import numpy as np
from model import two_layer_net_a3c
import time

# flags for general run
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Flags for experiment settings
tf.app.flags.DEFINE_string("env", "Pong-v0", "Gym environment to run")
tf.app.flags.DEFINE_float("gamma", .99, "Discount factor")
tf.app.flags.DEFINE_integer("num_episodes", 2,
                            "Number of episodes to run before updating")
tf.app.flags.DEFINE_integer("num_steps", 1000000, "Maximum number of updates")
tf.app.flags.DEFINE_boolean("diff_frame", True, "Whether to take difference of frames")
tf.app.flags.DEFINE_boolean("preprocess", True, "Whether to preprocess input")
tf.app.flags.DEFINE_string("actions", "2,3", "action space as comma separated strings")
tf.app.flags.DEFINE_integer("inp_dim", 80*80, "Input dimension for problem")
tf.app.flags.DEFINE_integer("out_dim", 2, "Output dimension for problem")

FLAGS = tf.app.flags.FLAGS

def main(arg):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        server.join()
    else:
        is_chief = (task_index == 0)

        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            env, env_name = gym.make(FLAGS.env), FLAGS.env
            gamma = FLAGS.gamma
            diff_frame, check_process = FLAGS.diff_frame, FLAGS.preprocess
            pre = preprocess if check_process else None 
            actions = np.array(map(int, FLAGS.actions.split(',')))
            n_actions = actions.size

            model = two_layer_net_a3c(FLAGS.inp_dim, FLAGS.out_dim)

            state_ph, actions_ph, rewards_ph = model['states'],model['actions'],model['rewards']
            probs, value = model['probs'], model['value']
            optimize = model['optimize']
            init_op, summary_op = model['init_op'], model['summary_op']
            running_reward_ph, saver = model['running_reward'], model['saver']
            episode_reward_ph = model['episode_reward']

        logdir = "./" + env_name + "_train_logs/"
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=None,  # disable summary thread;crashes
                                 saver=saver,
                                 save_model_secs=200)

        with sv.managed_session(server.target) as sess:
            print("Waiting for 5 seconds")

            writer = tf.train.SummaryWriter(logdir, graph=tf.get_default_graph())
            step = 0
            avg_reward  = None
            n_e = 0
            terminal = False
            s_t = None
            ep_rwd = 0

            while not sv.should_stop() and n_e < 2000:
                o_n, a_n, r_n, s_t, terminal = run_episode_a3c(env, 
                                                               model,
                                                               sess, 
                                                               actions, 
                                                               s_t, 
                                                               terminal,
                                                               diff_frame=diff_frame,
                                                               preprocess=pre)
                ep_rwd += np.sum(r_n)

                if terminal:
                    R = 0
                else:
                    R = sess.run(value, feed_dict={state_ph:[s_t]})[0][0]

                disc_r = discount_rewards(r_n, 0.99, env_name, initial=R)
                #disc_r -= np.mean(disc_r)
                #disc_r /= np.std(disc_r)

                sess.run(optimize, feed_dict={
                                    state_ph: o_n,
                                    actions_ph: np.eye(n_actions)[a_n],
                                    rewards_ph: disc_r })

                if terminal:
                    avg_reward = ep_rwd if avg_reward is None else 0.99*avg_reward + 0.01*ep_rwd
                    print "Episode %d, Avg Reward %1.3f and Episode Reward %1.3f" % (n_e, avg_reward, ep_rwd)
                    
                    summary = sess.run(summary_op, feed_dict={
                                                    running_reward_ph: avg_reward,
                                                    episode_reward_ph: ep_rwd})
                    ep_rwd = 0
                    terminal = False
                    s_t = None # reset
                    n_e += 1
                    writer.add_summary(summary, n_e)

                step += 1

            print('stopped')
            sv.stop()
            writer.flush()

if __name__ == "__main__":
    tf.app.run()




'''
Asynchronous distributed Policy Gradients
'''
import tensorflow as tf
from pg import run_episode, discount_rewards
import gym
import numpy as np
from model import two_layer_net
from util import cluster_config

# Flags for defining the tf.train.ClusterSpec
# flags if run on aws
tf.app.flags.DEFINE_integer("aws", "0", "Running code on aws or not")
tf.app.flags.DEFINE_string("config_path", "./tmp/cfg",
              "path for cluster config files")
tf.app.flags.DEFINE_integer("node_id", 0, "Index of node in the cluster")
tf.app.flags.DEFINE_integer("n_ps", 1 , "Number of param servers in cluster")
tf.app.flags.DEFINE_integer("n_nodes", 3 , "Number of nodes in cluster")

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

FLAGS = tf.app.flags.FLAGS

def main(arg):
    if not FLAGS.aws:
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server for the local task.
        job_name = FLAGS.job_name
        task_index = FLAGS.task_index
        server = tf.train.Server(cluster,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index)

    else:
        config = cluster_config(FLAGS)
        cluster = tf.train.ClusterSpec({"ps": config['ps_hosts'],
                    "worker": config['worker_hosts']})
        job_name = config['job']
        task_index = config['task_id']
        server = tf.train.Server(cluster,
                 job_name=config['job'],
                 task_index=config['task_id'])

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        is_chief = (task_index == 0)

        # Assigns ops to the local worker by default.
        # note: this automatically sets the device on which ops/vars are stored
        # ops are stored on the local worker running this code, vars are stored
        # on param server - so each worker has network copy but actual weight
        # values are shared through paramserver
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            env = gym.make(FLAGS.env)
            # !!!PONG specific: only use up and down
            actions = np.array([2, 3])  # env.action_space
            n_actions = actions.size

            running_reward = tf.Variable(0., name="running_reward")
            tf.scalar_summary("Running Reward", running_reward)
            summary_op = tf.merge_all_summaries()
            model = two_layer_net(80*80, 200, 2)

            S, A, Adv = model['input_ph'], model['actions_ph'], model['advantage_ph']
            net, optimizer, loss = model['net'], model['optimizer'], model['loss']
            saver, init_op, global_step = model['saver'], model['init_op'], model['global_step']


        # Create a "supervisor", which oversees the training process.
        logdir = "./tmp/train_logs/"
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=None,  # disable summary thread;crashes
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=10)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print "waiting for 10 secs to get a checkpoint saved"
            import time
            time.sleep(13)

            writer = tf.train.SummaryWriter(logdir, graph=sess.graph)
            step = 0  # global training steps
            avg_reward = None  # running average episode reward
            local_step = 0  # local training steps performed
            n_e = 0
            # Loop until the supervisor shuts down or max steps have completed.
            while not sv.should_stop() and step < FLAGS.num_steps:
                # get data by interacting with env using current policy
                obs, acts, returns = None, None, None
                rwds = 0
                for e in range(FLAGS.num_episodes):
                    # get a single episode
                    o_n, a_n, r_n = run_episode(env,model,server,logdir,actions, n_e)
                    # get episode discounted return
                    disc_r = discount_rewards(r_n, FLAGS.gamma)
                    disc_r -= np.mean(disc_r)
                    disc_r /= np.std(disc_r)
                    # store results
                    r = np.sum(r_n)
                    rwds += r
                    avg_reward = r if avg_reward is None\
                        else .99 * avg_reward + .01 * r
                    obs = o_n if obs is None else np.append(obs, o_n, axis=0)
                    acts = a_n if acts is None else np.append(acts, a_n)
                    returns = disc_r if returns is None else np.append(returns,
                                                                       disc_r)
                    n_e +=1

                # feed trajectories to pg training
                _, step_loss, step, summary = sess.run([optimizer, loss, global_step, summary_op],
                                              feed_dict={S: obs,
                                                         A: acts,
                                                         Adv: returns,
                                                         running_reward: avg_reward})
                writer.add_summary(summary, n_e)
                num = FLAGS.num_episodes
                print('step %d, rew %1.3f, avg reward for %d episodes is %1.3f' % (local_step, avg_reward, num, rwds/num))
                local_step += 1
            # Ask for all the services to stop.
            print('stopped')
            sv.stop()

if __name__ == "__main__":
    tf.app.run()

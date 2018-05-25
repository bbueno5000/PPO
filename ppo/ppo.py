"""

Proximal Policy Optimization (PPO)

Contains an implementation of PPO as described here: https://arxiv.org/abs/1707.06347

"""
import agents.environment as environ
import logging
import numpy as np
import os
import ppo.renderthread as ppo_thread
import ppo.models as ppo_models
import ppo.trainer as ppo_tnr
import shutil
import tensorflow as tf
import time

if __name__ == '__main__':
    # ALGORITHM PARAMETERS
    batch_size = 128                           # How many experiences per gradient descent update step [default: 64]
    beta = 2.5e-3                              # Strength of entropy regularization [default: 2.5e-3]
    buffer_size = batch_size * 32              # How large the experience buffer should be before gradient descent [default: 2048]
    epsilon = 0.2                              # Acceptable threshold around ratio of old and new policy probabilities [default: 0.2]
    gamma = 0.99                               # Reward discount rate [default: 0.99]
    hidden_units = 128                         # Number of units in hidden layer [default: 64]
    lambd = 0.95                               # Lambda parameter for GAE [default: 0.95]
    learning_rate = 4e-5                       # Model learning rate [default: 3e-4]
    max_steps = 15e6                           # Maximum number of steps to run environment [default: 1e6]
    normalize_steps = 0                        # Activate state normalization for this many steps and freeze statistics afterwards
    num_epoch = 10                             # Number of gradient descent steps per batch of experiences [default: 5]
    num_layers = 1                             # Number of hidden layers between state/observation and outputs [default: 2]
    time_horizon = 2048                        # How many steps to collect per agent before adding to buffer [default: 2048]
    # GENERAL PARAMETERS
    keep_checkpoints = 5                       # How many model checkpoints to keep [default: 5]
    load_model = True                          # Whether to load the model or randomly initialize [default: False]
    model_path = '.\\models\\working'          # The sub-directory name for model and summary statistics
    record = True                              # save recordings of episodes
    render = True                              # render environment to display progress
    summary_freq = buffer_size * 5             # Frequency at which to save training statistics [default: 10000]
    summary_path = '.\\ppo_summary'            # The sub-directory name for model and summary statistics
    save_freq = summary_freq                   # Frequency at which to save model [default: 50000]
    train_model = False                        # Whether to train model, or only run inference [default: False]
    env_name = 'VerticalLanding-v0'
    env = environ.GymEnvironment(env_name, log_path='.\\ppo_log', skip_frames=6)
    env_render = environ.GymEnvironment(env_name, log_path='.\\ppo_log_render', render=True, record=record)
    fps = env_render.env.metadata.get('video.frames_per_second', 30)
    logging.info(str(env))
    brain_name = env.external_brain_names[0]
    tf.reset_default_graph()
    ppo_model = ppo_models.create_agent_model(env,
                                              learning_rate,
                                              hidden_units,
                                              epsilon,
                                              beta,
                                              max_steps,
                                              normalize_steps,
                                              num_layers)
    is_continuous = env.brains[brain_name].action_space_type == 'continuous'
    use_observations = False
    use_states = True
    if not load_model:
        shutil.rmtree(summary_path, ignore_errors=True)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    tf.set_random_seed(np.random.randint(1024))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=keep_checkpoints)
    with tf.Session() as sess:
        # Instantiate model parameters
        if load_model:
            logging.info('model:loading')
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                logging.error('model could not be found:{}'.format(model_path))
            saver.restore(sess, ckpt.model_checkpoint_path)    # pylint:disable=E0611
        else:
            sess.run(init)
        steps, last_reward = sess.run([ppo_model.global_step, ppo_model.last_reward])
        summary_writer = tf.summary.FileWriter(summary_path)
        info = env.reset()[brain_name]
        trainer = ppo_tnr.Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, train_model)
        trainer_monitor = ppo_tnr.Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, False)
        render_started = False
        while steps <= max_steps or not train_model:
            if env.global_done:
                info = env.reset()[brain_name]
                trainer.reset_buffers(info, total=True)
            # decide and take an action
            if train_model:
                info = trainer.take_action(info, env, brain_name, steps, normalize_steps)
                trainer.process_experiences(info, time_horizon, gamma, lambd)
            else:
                time.sleep(1)
            if len(trainer.training_buffer['actions']) > buffer_size and train_model:
                if render:
                    renderthread.pause()
                logging.info('optimization:started')
                t = time.time()
                # perform gradient descent with experience buffer
                trainer.update_model(batch_size, num_epoch)
                logging.info('optimization:finished')
                logging.info('optimization:{:.1f} seconds'.format(float(time.time() - t)))
                if render:
                    renderthread.resume()
            if steps % summary_freq == 0 and steps != 0 and train_model:
                # write training statistics to tensorboard.
                trainer.write_summary(summary_writer, steps)
            if steps % save_freq == 0 and steps != 0 and train_model:
                # save Tensorflow model
                ppo_models.save_model(sess, saver, model_path, steps)
            if train_model:
                steps += 1
                sess.run(ppo_model.increment_step)
                if len(trainer.stats['cumulative_reward']) > 0:
                    mean_reward = np.mean(trainer.stats['cumulative_reward'])
                    sess.run(ppo_model.update_reward, feed_dict={ppo_model.new_reward: mean_reward})
                    last_reward = sess.run(ppo_model.last_reward)
            if not render_started and render:
                renderthread = ppo_thread.RenderThread(sess,
                                                       trainer_monitor,
                                                       env_render,
                                                       brain_name,
                                                       normalize_steps,
                                                       fps)
                renderthread.start()
                render_started = True
        # final save Tensorflow model
        if steps != 0 and train_model:
            ppo_models.save_model(sess, saver, model_path, steps)
    env.close()
    ppo_models.export_graph(model_path, env_name)
    os.system('shutdown')

import time

from TrainerUtils import write_video, compute_avg_return

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tqdm import tqdm
import imageio


class BaseTrainer:

    def __init__(self):
        self.environmnet_name = self.get_environment_name()
        self.training_information = self.get_training_information()

        self.num_atoms = self.training_information['num_atoms']
        self.learning_rate = self.training_information['learning_rate']
        self.min_q_value = self.training_information['min_q_value']
        self.max_q_value = self.training_information['max_q_value']
        self.n_step_update = self.training_information['n_step_update']
        self.gamma = self.training_information['gamma']
        self.num_eval_episodes = self.training_information['num_eval_episodes']
        self.replay_buffer_capacity = self.training_information['replay_buffer_capacity']
        self.initial_collect_steps = self.training_information['initial_collect_steps']
        self.batch_size = self.training_information['batch_size']
        self.num_iterations = self.training_information['num_iterations']
        self.collect_steps_per_iteration = self.training_information['collect_steps_per_iteration']
        self.log_interval = self.training_information['log_interval']
        self.eval_interval = self.training_information['eval_interval']
        self.render_interval = self.training_information['render_interval']

        self.train_py_env = suite_gym.load(self.environmnet_name)
        self.eval_py_env = suite_gym.load(self.environmnet_name)

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.categorical_q_net = self.get_categorical_q_net()
        self.optimizer = self.get_optimizer()
        self.train_step_counter = tf.Variable(0)
        self.agent = self.get_agent()
        self.agent.initialize()

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                             self.train_env.action_spec())

        self.replay_buffer = self.get_replay_buffer()

    def get_replay_buffer(self):
        return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
                                                              batch_size=self.train_env.batch_size,
                                                              max_length=self.replay_buffer_capacity)

    def get_agent(self):
        return categorical_dqn_agent.CategoricalDqnAgent(
                    self.train_env.time_step_spec(),
                    self.train_env.action_spec(),
                    # epsilon_greedy=0.8,
                    categorical_q_network=self.categorical_q_net,
                    optimizer=self.optimizer,
                    min_q_value=self.min_q_value,
                    max_q_value=self.max_q_value,
                    n_step_update=self.n_step_update,
                    td_errors_loss_fn=common.element_wise_squared_loss,
                    gamma=self.gamma,
                    train_step_counter=self.train_step_counter)

    def get_optimizer(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

    def get_categorical_q_net(self):
        preprocessing_layers = self.get_preprocessing_layers()
        fc_layer_params = self.get_fc_layer_params()

        return categorical_q_network.CategoricalQNetwork(
                    self.train_env.observation_spec(),
                    self.train_env.action_spec(),
                    num_atoms=self.num_atoms,
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
                    fc_layer_params=fc_layer_params
                )

    def get_preprocessing_layers(self):
        return {'image': Sequential(layers=[
                                             Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish'),
                                             Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish'),
                                             MaxPooling2D(pool_size=(2,2)),
                                             Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish'),
                                             Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish'),
                                             MaxPooling2D(pool_size=(2,2)),
                                             Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish'),
                                             Flatten()
                                           ], name='image'),
                'location': Sequential(layers=[
                                                Dense(12, activation='swish'),
                                                Dense(12, activation='swish'),
                                                Flatten()
                                              ], name='location')}

    def get_fc_layer_params(self):
        raise NotImplementedError('get_fc_layer_params must be implemented in a child class')

    def get_environment_name(self):
        raise NotImplementedError('get_environment_name must be implemented in a child class')

    def get_training_information(self):
        info = {
            'num_iterations': 1000000,
            'initial_collect_steps': 1000,
            'collect_steps_per_iteration': 1,
            'replay_buffer_capacity': 1000000,
            'batch_size': 128,
            'learning_rate': 1e-3,
            'gamma': 0.999,
            'num_atoms': 51,
            'min_q_value': -5000,
            'max_q_value': 100,
            'n_step_update': 2,
            'num_eval_episodes': 10,
            'eval_interval': 2000,
            'render_interval': 10000,
            'log_interval': 2000
        }

        return info

    def collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def train(self):
        print(f'average score for random agent: {compute_avg_return(self.eval_env, self.random_policy, self.num_eval_episodes)}')

        print('Gathering random examples...')
        for _ in tqdm(range(self.initial_collect_steps), mininterval=1.0):
            self.collect_step(self.train_env, self.random_policy)

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size,
            num_steps=self.n_step_update + 1).prefetch(3)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        print('Evaluating agent before training...')
        avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        print(avg_return)
        returns = [avg_return]

        print('Starting to train...')

        for _ in tqdm(range(self.num_iterations), mininterval=5.0):
            start = time.time()

            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(self.collect_steps_per_iteration):
                self.collect_step(self.train_env, self.agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience)

            step = self.agent.train_step_counter.numpy()

            end = time.time()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}: round'.format(step, train_loss.loss, end - start))

            if step % self.eval_interval == 0:
                avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                returns.append(avg_return)

            if step % self.render_interval == 0 and step > 0:
                write_video(self.eval_env, self.eval_py_env, self.agent, video_filename=f'render_{step}.mp4', num_rounds=5)

        print('returns:', returns)

        print('rendering a sample video')

        imageio.plugins.ffmpeg.download()

        write_video(self.eval_env, self.eval_py_env, self.agent, video_filename='final.mp4', num_rounds=25)


class EasyTrainer(BaseTrainer):

    def __init__(self):
        BaseTrainer.__init__(self)

    def get_environment_name(self):
        return 'primal_env:Primal-v0'

    def get_fc_layer_params(self):
        return 100, 100, 100

    def get_training_information(self):
        info = super().get_training_information()
        info['num_iterations'] = 50000
        info['gamma'] = 0.99
        info['render_interval'] = 1

        return info


class SingleTrainer(BaseTrainer):

    def __init__(self):
        BaseTrainer.__init__(self)

    def get_environment_name(self):
        return 'primal_env:Primal-Single-v0'

    def get_fc_layer_params(self):
        return 128, 128, 128, 128, 64

    def get_training_information(self):
        info = super().get_training_information()
        info['num_iterations'] = 150000
        info['gamma'] = 0.99

        return info

class FourAndFourTrainer(SingleTrainer):

    def __init__(self):
        SingleTrainer.__init__(self)

    def get_environment_name(self):
        return 'primal_env:Primal-FourEnemyFourAlly-v0'


class FourFourFiftyTrainer(SingleTrainer):

    def __init__(self):
        SingleTrainer.__init__(self)

    def get_environment_name(self):
        return 'primal_env:Primal-FourEnemyFourAllyFiftyWall-v0'

    def get_fc_layer_params(self):
        return 512, 512, 256, 256, 128

    def get_training_information(self):
        info = super().get_training_information()
        info['num_iterations'] = 200000
        info['gamma'] = 0.1

        return info



if __name__ == '__main__':
    trainer = FourFourFiftyTrainer()
    trainer.train()

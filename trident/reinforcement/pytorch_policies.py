import builtins
import copy
import gc
import math
import random
import time
import uuid
import os
import sys
import threading
import subprocess
import webbrowser
from time import sleep
from copy import deepcopy
from itertools import count
from typing import List, Tuple, Optional, Union, Callable, Any, Iterable, Mapping, TypeVar, Dict
import numpy as np
import torch
from torch.distributions import Bernoulli, Categorical

from trident import context
from trident.backend.common import *
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_ops import *
from trident.backend.tensorspec import TensorSpec, ObjectType
from trident.data.image_common import image_backend_adaption
from trident.layers.pytorch_initializers import kaiming_normal, orthogonal
from trident.misc.visualization_utils import loss_metric_curve
from trident.optims.pytorch_trainer import Model, MuiltiNetwork
from trident.reinforcement.utils import ReplayBuffer, Rollout, ActionStrategy
from trident.loggers.history import HistoryBase
from trident.context import split_path, make_dir_if_need, sanitize_path
import_or_install('gym')
import gym

ctx = context._context()
_backend = ctx.get_backend()
working_directory = ctx.working_directory

__all__ = ['PolicyBase', 'ActorCriticPolicy', 'DqnPolicy', 'PGPolicy', 'A2CPolicy', 'PPOPolicy']


class PolicyBase(Model):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, action_strategy=None, gamma=0.99, use_experience_replay=False,
                 replay_unit='step', memory_length: int = 10000,
                 name=None) -> None:
        super().__init__()
        self.network = network
        if name is not None:
            self.network._name = name

        self.env = env
        self.state = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.agent_id = uuid.uuid4().node
        self.gamma = gamma
        self.use_experience_replay = use_experience_replay
        self.memory = None
        if self.use_experience_replay:
            self.memory = ReplayBuffer(memory_length)
        else:
            self.memory = ReplayBuffer(1)

        if replay_unit not in ['step', 'nstep', 'episode']:
            raise ValueError('Only [step,episode] are valid unit options.')
        self.replay_unit = replay_unit
        self.name = name
        self.action_strategy = action_strategy

        self.train_steps = -1
        self.frame_steps = -1
        self.rollout = Rollout()
        self.setting_network()

    def setting_network(self):
        super()._initial_graph(inputs=self.get_observation(), output=deepcopy(self.network))

    def get_observation(self):
        self.do_on_get_observation_start()
        state = None
        if hasattr(self.env, 'state'):
            state = expand_dims(self.data_preprocess(self.env.state), 0)
        elif hasattr(self.env, 'screen'):
            state = expand_dims(self.data_preprocess(self.env.screen), 0)
        elif 'observation' in self.env.metadata['render.modes']:
            state = expand_dims(self.data_preprocess(self.env.render('observation')), 0)
        else:
            state = expand_dims(self.data_preprocess(self.env.render('rgb_array')), 0)
        self.do_on_get_observation_end(state)
        return state

    def select_action(self, state, model_only=False, **kwargs):
        self.do_on_select_action_start()
        return self.env.action_space.samples()

    def get_rewards(self, action):
        self.do_on_get_rewards_start()
        return self.env.step(action)

    def experience_replay(self, batch_size):
        return NotImplemented

    def collect_samples(self, min_replay_samples, need_render=False) -> bool:

        if self.memory is None:
            self.memory = ReplayBuffer(10000)
        progress_inteval = int(min_replay_samples / 50) * 5
        self.state_pool = []
        self.reward_pool = []
        self.action_pool = []

        for i_episode in range(min_replay_samples):
            self.env.reset()
            state = self.get_observation()
            for t in count():
                action = self.select_action(state,
                                            model_only=True if self.action_strategy == ActionStrategy.OnPolicy else False)
                _observation, reward, done, info = self.get_rewards(action)
                if need_render:
                    self.env.render()
                next_state = None
                if not done:
                    next_state = self.get_observation()
                if self.replay_unit == 'step':
                    self.memory.push(state, action, next_state, reward)
                    if len(self.memory) < min_replay_samples and len(self.memory) % progress_inteval == 0:
                        print("Replay Samples:{0}".format(len(self.memory)))
                    if len(self.memory) == min_replay_samples:
                        # n1 = self.action_logs['model'][0]
                        # n2 = self.action_logs['model'][1]
                        # n3 = self.action_logs['random'][0]
                        # n4 = self.action_logs['random'][1]
                        # print('model: 0:{0} 1:{1}  random: 0:{2} 1:{3}  random: {4}'.format(float(n1) / (n1 + n2), float(n2) / (n1 + n2), float(n3) / (n3 + n4),
                        #                                                                                       float(n4) / (n3 + n4), float(n3 + n4) / builtins.max(n1 + n2 + n3
                        #                                                                                       + n4,1)))
                        #
                        # self.action_logs = OrderedDict()
                        # self.action_logs['model'] = OrderedDict()
                        # self.action_logs['random'] = OrderedDict()
                        # self.action_logs['model'][0] = 0
                        # self.action_logs['model'][1] = 0
                        # self.action_logs['random'][0] = 0
                        # self.action_logs['random'][1] = 0
                        return True
                elif self.replay_unit == 'episode':
                    self.state_pool.append(state)
                    self.action_pool.append(action)
                    self.reward_pool.append(reward)
                    if done:
                        self.memory.push(self.state_pool, self.action_pool, None, self.reward_pool)
                        if len(self.memory) < min_replay_samples and len(self.memory) % progress_inteval == 0:
                            print("Replay Samples:{0}".format(len(self.memory)))
                        self.state_pool = []
                        self.action_pool = []
                        self.reward_pool = []

                        if len(self.memory) == min_replay_samples:
                            return True
                        break
                state = next_state
                if done:
                    break

        return False

    def push_into_memory_criteria(self, *args, **kwargs) -> bool:
        return True

    def episode_complete_criteria(self, *args, **kwargs) -> bool:
        return False

    def task_complete_criteria(self, *args, **kwargs) -> bool:
        return False

    def calculate_discounted_returns(self, *args, **kwargs):
        return NotImplemented

    def save_or_sync_weights(self):
        self.save_model(save_path=self.training_context['save_path'])

    def soft_update(self, trained_model, target_model, tau=1e-2):
        ''' Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            trained_model (Trident model or layer): weights will be copied from
            target_model  (Trident model or layer): weights will be copied to
            tau (float): interpolation parameter

        Returns:

        '''
        if isinstance(target_model, (Model, Layer)) and isinstance(trained_model, (Model, Layer)):
            target_model_parameters = target_model._model.parameters() if isinstance(target_model,
                                                                                     Model) else target_model.parameters()
            trained_model_parameters = trained_model._model.parameters() if isinstance(trained_model,
                                                                                       Model) else trained_model.parameters()
            for target_param, trained_param in zip(target_model_parameters, trained_model_parameters):
                target_param.data.copy_(tau * trained_param.data + (1.0 - tau) * target_param.data)
            return target_model
        else:
            raise ValueError('target_model and trained_model should be trident Model or Layer.')

    def calculate_gae(self, values, next_values, rewards, dones, gamma: float, gae_lambda: float):
        R = [0] * len(rewards)
        delta = rewards + next_values * gamma - values
        m = (1.0 - dones) * (gamma * gae_lambda)
        gae = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            R[i] = gae
        return R

    def training_model(self, current_episode=0, current_step=0, num_episodes=100, train_timing=None, done=False,
                       batch_size=1, repeat_train=1):

        is_collect_data = False
        for i in range(repeat_train):
            self.train_steps += 1
            data = None
            if self.use_experience_replay:
                data = self.experience_replay(batch_size)
            else:
                data = self.memory.memory[0]
            self.calculate_discounted_returns(*data)
            self.training_context['skip_generate_output'] = True
            if 'step' in train_timing:
                current_step = current_step * repeat_train + i
                is_collect_data = True

            elif 'episode' in train_timing:
                current_step = i

            super(PolicyBase, self).train_model(self.training_context['train_data'], self.training_context['test_data'],
                                                current_epoch=current_episode,
                                                current_batch=current_step,
                                                total_epoch=num_episodes,
                                                done=(done and not 'episode' in train_timing) or (
                                                        current_step == repeat_train - 1 and 'episode' in train_timing),
                                                is_collect_data=True,
                                                is_print_batch_progress=False,
                                                is_print_epoch_progress=False,
                                                log_gradients=False, log_weights=False,
                                                accumulate_grads=(
                                                                         current_step * repeat_train + 1) % self.accumulation_steps != 0)
            self.save_or_sync_weights()

    def play(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, training=True,
             train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
             need_render=True, **kwargs):
        if train_timing not in ['on_episode_start', 'on_step_end', 'on_step_start']:
            raise ValueError('Only on_episode_start,on_step_end are valid  train_timing options')

        if training:
            self._model.train()
        else:
            self._model.eval()
        if self.use_experience_replay:
            self.collect_samples(min_replay_samples=min_replay_samples)
        else:
            self.collect_samples(min_replay_samples=1, need_render=True if self.replay_unit == 'episode' else False)
            print('start train....')

        self.total_reward = 0
        self.t = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.total_epoch = num_episodes
        self.steps = 0
        self.frame_steps = -1
        for i_episode in range(self.total_epoch):

            if training and train_timing == 'on_episode_start' and self.current_epoch % train_every_nstep == 0:
                self.training_model(self.current_epoch, 0, num_episodes=self.total_epoch, repeat_train=repeat_train,
                                    train_timing=train_timing, batch_size=batch_size)

            self.env.reset()
            self.total_rewards = 0
            state = self.get_observation()

            for t in count():
                self.frame_steps += 1
                self.t = t
                # # Train on_step_start
                # if training and train_timing == 'on_step_start' and t % train_every_nstep == 0:
                #     self.training_model(i_episode, t,num_episodes=num_episodes, repeat_train=repeat_train, batch_size=batch_size)

                action = self.select_action(state, model_only=True)
                self.do_on_select_action_end(action)

                observation, reward, done, info = self.get_rewards(action)
                self.do_on_get_rewards_end(observation, reward, done, info)

                self.total_rewards += reward

                next_state = self.get_observation() if not done else None

                if need_render:
                    self.env.render()
                if self.replay_unit == 'step':
                    if self.push_into_memory_criteria(state, action, next_state, reward) or done:
                        self.memory.push(state, action, next_state, reward)
                elif self.replay_unit == 'episode':
                    self.rollout.collect('state', state)
                    self.rollout.collect('action', action)
                    self.rollout.collect('reward', reward)

                    if done:
                        if self.push_into_memory_criteria(self.state_pool, self.action_pool, None, self.reward_pool):
                            self.memory.push(self.state_pool, self.action_pool, None, self.reward_pool)
                        self.rollout.reset()

                complete = self.episode_complete_criteria()
                # Train on_step_end
                if training and train_timing == 'on_step_end' and (
                        (self.self.frame_steps + 1) % train_every_nstep == 0 or done):
                    self.training_model(i_episode, t, num_episodes=num_episodes, done=done or complete,
                                        train_timing=train_timing, repeat_train=repeat_train, batch_size=batch_size)

                state = next_state
                if done or complete:
                    self.env.reset()
                    if training and train_timing == 'on_episode_end' and i_episode % train_every_nstep == 0:
                        self.training_model(i_episode, 0, num_episodes=num_episodes, repeat_train=repeat_train,
                                            train_timing=train_timing, batch_size=batch_size)

                    self.epoch_metric_history.collect('rewards', i_episode, float(self.total_rewards))
                    self.epoch_metric_history.collect('t', i_episode, float(t + 1))
                    if self.use_experience_replay:
                        self.epoch_metric_history.collect('replay_buffer_utility', i_episode,
                                                          float(len(self.memory)) / self.memory.capacity)

                    self.do_on_epoch_end()
                    if print_progess_frequency == 1 or (
                            i_episode > 0 and (i_episode + 1) % print_progess_frequency == 0):
                        self.print_epoch_progress()

                    if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                        loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history,
                                          metrics_names=list(self.epoch_metric_history.keys()), calculate_base='epoch',
                                          imshow=True)

                    if self.task_complete_criteria():
                        self.save_model(save_path=self.training_context['save_path'])
                        print('episode {0} meet task complete criteria, training finish! '.format(i_episode))
                        return True

                    break

        print('Complete')
        self.env.render()
        self.env.close()

    def learn(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5,
              train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
              need_render=True, **kwargs):
        if ctx.enable_tensorboard:
            for k, v in self.training_context.items():
                if isinstance(v, HistoryBase):
                    v.training_name = self.name

            self.training_context['training_name'] = self.name
            self.training_context['summary_writer'] = ctx.summary_writer

            t1 = threading.Thread(target=launchTensorBoard, args=([]))
            t1.setDaemon(True)
            t1.start()
            open_browser('http://{0}:{1}/'.format(ctx.tensorboard_server, ctx.tensorboard_port), 5)
        if ctx.enable_mlflow:
            ctx.mlflow_logger.start_run(run_id=self.execution_id)
            open_browser('http://{0}:{1}/'.format(ctx.mlflow_server, ctx.mlflow_port), 5)

        self.training_context['repeat_train'] = repeat_train
        self.training_context['train_timing'] = train_timing
        self.training_context['train_every_nstep'] = train_every_nstep
        self.training_context['min_replay_samples'] = min_replay_samples
        self.training_context['batch_size'] = batch_size

        self.do_on_training_start()

        try:
            self.play(num_episodes=num_episodes, batch_size=batch_size, min_replay_samples=min_replay_samples,
                      print_progess_frequency=print_progess_frequency, training=True,
                      train_timing=train_timing, train_every_nstep=train_every_nstep,
                      repeat_train=repeat_train, need_render=need_render)
        except Exception as e:
            print(e)
            PrintException()

    def resume(self, num_episodes=3000, **kwargs):
        pass

    @property
    def preprocess_flow(self):
        return self._preprocess_flow

    @preprocess_flow.setter
    def preprocess_flow(self, value):
        self._preprocess_flow = value
        objecttype = None
        if isinstance(self.model.input_spec, TensorSpec):
            objecttype = self.model.input_spec.object_type
        # super()._initial_graph(inputs=to_tensor(self.get_observation()).repeat_elements(2, 0), output=deepcopy(self.network))
        self.setting_network()
        if objecttype is not None:
            self.inputs.value_list[0].object_type = objecttype
            self.model.input_spec.object_type = objecttype

        self.env.reset()

    def data_preprocess(self, img_data):
        if self._model is not None:
            self._model.input_spec.object_type = ObjectType.rgb
        if not hasattr(self, '_preprocess_flow') or self._preprocess_flow is None:
            self._preprocess_flow = []
        if img_data.ndim == 4:
            return to_tensor(to_numpy([self.data_preprocess(im) for im in img_data]))
        if len(self._preprocess_flow) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self._preprocess_flow:
                if self._model is not None and self.signature is not None and len(
                        self.signature) > 1 and self._model.input_spec is not None:
                    img_data = fc(img_data, spec=self._model.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)
            if self._model.input_spec is None:
                self._model.input_spec = TensorSpec(
                    shape=tensor_to_shape(to_tensor(img_data), need_exclude_batch_axis=True, is_singleton=True),
                    object_type=ObjectType.rgb,
                    name='input')

                self.input_shape = self._model.input_spec.shape[1:]

            return img_data
        else:
            return img_data

    def do_on_get_observation_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_get_observation_start(self.training_context)

    def do_on_get_observation_end(self, observation):
        for callback in self.training_context['callbacks']:
            callback.on_get_observation_end(self.training_context)

    def do_on_select_action_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_select_action_start(self.training_context)

    def do_on_select_action_end(self, action):
        for callback in self.training_context['callbacks']:
            callback.on_select_action_end(self.training_context)

    def do_on_get_rewards_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_get_rewards_start(self.training_context)

    def do_on_get_rewards_end(self, observation, reward, done, info):
        for callback in self.training_context['callbacks']:
            callback.on_get_rewards_end(self.training_context)

    def do_on_batch_end(self):
        self.training_context['time_batch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])
        if (self.training_context['steps'] + 1) % ctx.epoch_equivalent == 0:
            if self.warmup > 0 and self.warmup == (self.training_context['steps'] + 1) // ctx.epoch_equivalent:
                self.adjust_learning_rate(self.training_context['base_lr'])
                self.warmup = 0

    def with_tensorboard(self):
        make_dir_if_need(os.path.join(working_directory, 'Logs'))
        # check weather have tensorboard
        if get_backend() == 'pytorch':
            try:
                from trident.loggers.pytorch_tensorboard import SummaryWriter
                ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))

            except Exception as e:
                print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                print(e)
                PrintException()
        elif get_backend() == 'tensorflow':
            try:
                from trident.loggers.tensorflow_tensorboard import SummaryWriter
                ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))

            except Exception as e:
                print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                print(e)
                PrintException()
        return self

    def with_mlflow(self):
        from trident.loggers.mlflow_logger import MLFlowLogger
        ctx.try_enable_mlflow(MLFlowLogger())

        return self


def Critic(cls):
    """Actor Wrapper class that provides proxy access to an instance of actor model."""

    class Wrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.wrap = cls()

        def __call__(self, inputs: Union[Dict[TensorSpec, np.ndarray], np.ndarray], spec: TensorSpec = None, **kwargs):
            if self.rn % 5 > 0:
                return self.wrap(inputs, spec=spec, **kwargs)
            else:
                return inputs

        def __getattr__(self, name):
            return getattr(self.wrap, name)

    Wrapper.__name__ = Wrapper.__qualname__ = cls.__name__
    Wrapper.__doc__ = cls.__doc__
    return Wrapper


def Actor(cls):
    """Actor Wrapper class that provides proxy access to an instance of actor model."""

    class Wrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.wrap = cls()

        def __call__(self, inputs: Union[Dict[TensorSpec, np.ndarray], np.ndarray], spec: TensorSpec = None, **kwargs):
            if self.rn % 5 > 0:
                return self.wrap(inputs, spec=spec, **kwargs)
            else:
                return inputs

        def __getattr__(self, name):
            return getattr(self.wrap, name)

    Wrapper.__name__ = Wrapper.__qualname__ = cls.__name__
    Wrapper.__doc__ = cls.__doc__
    return Wrapper


class ActorCriticPolicy(MuiltiNetwork):
    """The base class for any RL policy.
    """

    def __init__(self, actor: Layer, critic: Layer, env: gym.Env, state_shape=None, accumulation_steps=1,
                 action_strategy=None, gamma=0.99, use_experience_replay=False, replay_unit='step',
                 memory_length: int = 10000,
                 name=None) -> None:
        super().__init__()

        self.rollout = Rollout()

        self.env = env
        self.state = None
        self.done = False
        self.accumulation_steps = accumulation_steps
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.agent_id = uuid.uuid4().node
        self.gamma = gamma
        self.use_experience_replay = use_experience_replay
        self.memory = None
        if self.use_experience_replay:
            self.memory = ReplayBuffer(memory_length)
        else:
            self.memory = ReplayBuffer(1)

        if replay_unit not in ['step', 'episode']:
            raise ValueError('Only [step,episode] are valid unit options.')
        self.replay_unit = replay_unit
        self.name = name
        self.action_strategy = action_strategy
        self.state_shape = state_shape
        if self.state_shape is not None:
            self.add_network('actor', input_shape=self.state_shape, output=actor)
            self.add_network('critic', input_shape=self.state_shape, output=critic)
        else:
            self.add_network('actor', inputs=self.get_observation(), output=actor)
            self.add_network('critic', inputs=self.get_observation(), output=critic)
        self.setting_network()
        self.train_steps = -1
        self.frame_steps = -1

    def setting_network(self):
        if isinstance(self._networks['actor'], Layer):
            self._networks['actor'] = Model(output=self._networks['actor'], inputs=self.get_observation())

        if isinstance(self._networks['critic'], Layer):
            self._networks['critic'] = Model(output=self._networks['critic'], inputs=self.get_observation())
        self._networks['critic'].training_context['model_name'] = 'critic'
        self._networks['actor'].training_context['model_name'] = 'actor'
        self._networks['critic'].training_context['skip_reset_total_loss'] = True
        self._networks['actor'].training_context['skip_reset_total_loss'] = True
        self._networks['critic'].training_context['skip_generate_output'] = True
        self._networks['actor'].training_context['skip_generate_output'] = True

    def get_observation(self):
        self.do_on_get_observation_start()
        state = None
        if hasattr(self.env, 'screen'):
            state = self.data_preprocess(self.env.screen)
        elif 'observation' in self.env.metadata['render.modes']:
            state = self.data_preprocess(self.env.render('observation'))
        else:
            state = self.data_preprocess(self.env.render('rgb_array'))
        if ndim(state) == 4:
            pass
        else:
            state = np.expand_dims(state, 0)
        self.do_on_get_observation_end(state)
        return state

    def select_action(self, state, model_only=False, **kwargs):
        self.do_on_select_action_start()
        return self.env.action_space.samples()

    def get_rewards(self, action):
        self.do_on_get_rewards_start()
        return self.env.step(action)

    def experience_replay(self, batch_size):
        return NotImplemented

    def collect_samples(self, min_replay_samples, need_render=False) -> bool:

        if self.memory is None:
            self.memory = ReplayBuffer(10000)
        progress_inteval = int(min_replay_samples / 50) * 5
        self.state_pool = []
        self.reward_pool = []
        self.action_pool = []

        for i_episode in range(min_replay_samples):
            self.env.reset()
            state = self.get_observation()
            for t in count():
                action = self.select_action(state,
                                            model_only=True if self.action_strategy == ActionStrategy.OnPolicy else False)
                _observation, reward, done, info = self.get_rewards(action)
                if need_render:
                    self.env.render()
                next_state = None
                if not done:
                    next_state = self.get_observation()
                if self.replay_unit == 'step':
                    self.memory.push(state, action, next_state, reward)
                    if len(self.memory) < min_replay_samples and len(self.memory) % progress_inteval == 0:
                        print("Replay Samples:{0}".format(len(self.memory)))
                    if len(self.memory) == min_replay_samples:
                        return True
                elif self.replay_unit == 'episode':
                    self.state_pool.append(state)
                    self.action_pool.append(action)
                    self.reward_pool.append(reward)
                    if done:
                        self.memory.push(self.state_pool, self.action_pool, None, self.reward_pool)
                        if len(self.memory) < min_replay_samples and len(self.memory) % progress_inteval == 0:
                            print("Replay Samples:{0}".format(len(self.memory)))
                        self.state_pool = []
                        self.action_pool = []
                        self.reward_pool = []

                        if len(self.memory) == min_replay_samples:
                            return True
                        break
                state = next_state
                if done:
                    break

        return False

    def push_into_memory_criteria(self, *args, **kwargs) -> bool:
        return True

    def episode_complete_criteria(self, *args, **kwargs) -> bool:
        return False

    def task_complete_criteria(self, *args, **kwargs) -> bool:
        return False

    def calculate_discounted_returns(self, *args, **kwargs):
        return NotImplemented

    def save_or_sync_weights(self):
        self.actor.save_model(save_path=self.actor.training_context['save_path'])
        self.critic.save_model(save_path=self.critic.training_context['save_path'])

    # def do_on_batch_end(self):
    #     self.training_context['time_batch_progress'] += (time.time() - self.training_context['time_batch_start'])
    #     self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])
    #     self.training_context['steps'] += 1
    #     for k in self._networks.keys():
    #         self._networks[k].training_context['time_batch_progress'] += (time.time() - self._networks[k].training_context['time_batch_start'])
    #         self._networks[k].training_context['time_epoch_progress'] += (time.time() - self._networks[k].training_context['time_batch_start'])
    #         self._networks[k].training_context['steps']=self.training_context['steps']

    def soft_update(self, trained_model, target_model, tau=1e-2):
        ''' Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            trained_model (Trident model or layer): weights will be copied from
            target_model  (Trident model or layer): weights will be copied to
            tau (float): interpolation parameter

        Returns:

        '''
        if isinstance(target_model, (Model, Layer)) and isinstance(trained_model, (Model, Layer)):
            target_model_parameters = target_model._model.parameters() if isinstance(target_model,
                                                                                     Model) else target_model.parameters()
            trained_model_parameters = trained_model._model.parameters() if isinstance(trained_model,
                                                                                       Model) else trained_model.parameters()
            for target_param, trained_param in zip(target_model_parameters, trained_model_parameters):
                target_param.data.copy_(tau * trained_param.data + (1.0 - tau) * target_param.data)
            return target_model
        else:
            raise ValueError('target_model and trained_model should be trident Model or Layer.')

    def calculate_gae(self, values, next_values, rewards, dones, gamma: float, gae_lambda: float, **kwargs):
        try:
            R = []
            # delta = to_tensor(rewards) + next_values * gamma - values
            # m = (1.0 - dones) * (gamma * gae_lambda)
            gae = 0.0
            # for i in range(len(rewards) - 1, -1, -1):
            #     gae = delta[i] + m[i] * gae
            #     R[i] = gae+values[i]
            for value, next_value, reward, done in list(zip(values, next_values, rewards, dones))[::-1]:
                gae = gae * gamma * gae_lambda
                gae = gae + reward + gamma * next_value.detach() * (1 - done) - value.detach()
                R.insert(0, gae + value)

            R = stack(R, axis=0)
            return R
        except Exception as e:
            print(e)

    def update_actor(self, current_episode=0, current_step=0, num_episodes=100, repeat_train=1, train_timing=None,
                     done=False, batch_size=1, accumulate_grads=False, **kwargs):
        is_collect_data = False
        traindata = OrderedDict()
        traindata['state'] = self.prev_screen
        for i in range(repeat_train):
            need_print_batch_progress = (self.train_steps + 1) % self.print_progess_frequency == 0
            self.actor.train_model(None, None,
                                   current_epoch=current_episode,
                                   current_batch=self.train_steps,
                                   total_epoch=self.total_epoch,
                                   done=self.done,
                                   is_collect_data=True,
                                   is_print_batch_progress=need_print_batch_progress,
                                   is_print_epoch_progress=self.done,
                                   log_gradients=False, log_weights=False,
                                   accumulate_grads=accumulate_grads)

    def update_critic(self, current_episode=0, current_step=0, num_episodes=100, repeat_train=1, train_timing=None,
                      done=False, batch_size=1, accumulate_grads=False, **kwargs):
        is_collect_data = False
        traindata = OrderedDict()
        traindata['state'] = self.prev_screen
        for i in range(repeat_train):
            need_print_batch_progress = (self.train_steps + 1) % self.print_progess_frequency == 0
            self.critic.train_model(None, None,
                                    current_epoch=current_episode,
                                    current_batch=self.train_steps,
                                    total_epoch=self.total_epoch,
                                    done=self.done,
                                    is_collect_data=True,
                                    is_print_batch_progress=need_print_batch_progress,
                                    is_print_epoch_progress=self.done,
                                    log_gradients=False, log_weights=False,
                                    accumulate_grads=accumulate_grads)

    def training_model(self, current_episode=0, current_step=0, num_episodes=100, train_timing=None, done=False,
                       batch_size=1, repeat_train=1, **kwargs):

        is_collect_data = False
        for i in range(repeat_train):
            self.train_steps += 1
            data = None

            self.calculate_discounted_returns(*data)
            self.training_context['skip_generate_output'] = True

            self.update_critic(current_episode=self.current_epoch,
                               current_step=self.train_steps,
                               num_episodes=num_episodes,
                               train_timing=train_timing,
                               done=done,
                               accumulate_grads=(self.train_steps * repeat_train + 1) % self.accumulation_steps != 0)
            self.update_cactor(current_episode=self.current_epoch,
                               current_step=self.train_steps,
                               num_episodes=num_episodes,
                               train_timing=train_timing,
                               done=done,
                               accumulate_grads=(self.train_steps * repeat_train + 1) % self.accumulation_steps != 0)
            self.save_or_sync_weights()

    def play(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, training=True,
             train_timing='on_step_end', train_every_nstep=1, repeat_train=1,
             need_render=True, **kwargs):
        if train_timing not in ['on_episode_start', 'on_episode_end', 'on_step_end']:
            raise ValueError('Only on_episode_start,on_step_end are valid  train_timing options')
        try:
            if training:
                self.actor.train()
                self.critic.train()
            else:
                self.actor.eval()
                self.critic.eval()
            if self.use_experience_replay:
                self.collect_samples(min_replay_samples=min_replay_samples)
            elif train_timing == 'on_episode_start':
                # self.collect_samples(min_replay_samples=1, need_render=True if self.replay_unit == 'episode' else False)
                print('start train....')
            self.print_progess_frequency = print_progess_frequency
            self.train_every_nstep = train_every_nstep
            self.total_epoch = num_episodes
            self.total_reward = 0
            self.t = 0
            self.current_epoch = 0
            self.train_steps = -1
            self.current_screen = None
            self.prev_screen = None

            if hasattr(self.env, 'recording_enabled'):
                self.env.recording_enabled = True
            for i_episode in range(num_episodes):
                self.current_epoch = i_episode
                self.current_batch = -1
                self.frame_steps = -1
                self.do_on_epoch_start()

                if training and train_timing == 'on_episode_start' and i_episode % train_every_nstep == 0:
                    self.training_model(i_episode, 0, num_episodes=num_episodes, repeat_train=repeat_train,
                                        train_timing=train_timing, batch_size=batch_size)
                # self.state = self.env.reset()

                self.total_rewards = 0
                self.done = False
                self.env.reset()
                state = self.get_observation()

                self.current_screen = state
                self.prev_screen = state
                for t in count():
                    self.frame_steps += 1
                    self.current_batch += 1
                    self.t = t

                    # # Train on_step_start
                    # if training and train_timing == 'on_step_start' and t % train_every_nstep == 0:
                    #     self.training_model(i_episode, t,num_episodes=num_episodes, repeat_train=repeat_train, batch_size=batch_size)

                    action = self.select_action(state, model_only=True)
                    self.do_on_select_action_end(action)

                    if hasattr(self, 'done') and self.done:
                        break
                    observation, reward, done, info = self.get_rewards(action)
                    self.do_on_get_rewards_end(observation, reward, done, info)

                    self.info = info
                    self.total_rewards += reward

                    next_state = observation if not done else zeros_like(observation)
                    self.prev_screen = self.current_screen
                    self.current_screen = next_state

                    if need_render:
                        self.env.render()

                    # Train on_step_end
                    complete = self.episode_complete_criteria()
                    if training and train_timing == 'on_step_end' and (
                            (len(self.rollout) + 1) % train_every_nstep == 0 or done) and self.frame_steps > 5:
                        try:
                            self.training_model(i_episode, t, num_episodes=num_episodes, done=done or complete,
                                                repeat_train=repeat_train, train_timing=train_timing)
                        except Exception as e:
                            print(e)

                    state = next_state

                    if done or complete:
                        # Train on_step_end
                        if training and train_timing == 'on_episode_end' and (
                                (self.frame_steps + 1) % train_every_nstep == 0 or done) and self.frame_steps > 5:
                            try:
                                self.training_model(i_episode, t, num_episodes=num_episodes, done=done or complete,
                                                    repeat_train=repeat_train, train_timing=train_timing)
                            except Exception as e:
                                print(e)

                        self.epoch_metric_history.collect('rewards', i_episode, float(self.total_rewards))
                        self.epoch_metric_history.collect('t', i_episode, float(t + 1))
                        if self.use_experience_replay:
                            self.epoch_metric_history.collect('replay_buffer_utility', i_episode,
                                                              float(len(self.memory)) / self.memory.capacity)

                        self.do_on_epoch_end()
                        if print_progess_frequency == 1 or ((i_episode + 1) % print_progess_frequency == 0):
                            self.print_epoch_progress()

                        # 定期繪製損失函數以及評估函數對時間的趨勢圖
                        if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                            keywords = ['reward', 'critic', 'actor', 'loss', 'score']
                            metrics_names = [k for k in self.epoch_metric_history.keys() if
                                             any([w in k.lower() for w in keywords]) or k.lower() == 't']
                            loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history,
                                              metrics_names=metrics_names, calculate_base='epoch',
                                              imshow=True)

                        if self.task_complete_criteria():
                            self.save_model(save_path=self.training_context['save_path'])
                            print('episode {0} meet task complete criteria, training finish! '.format(i_episode))
                            return True

            print('Complete')

        except Exception as e:
            print(e)
            PrintException()
            self.actor.save_model(save_path=self.actor.training_context['save_path'])
            self.critic.save_model(save_path=self.critic.training_context['save_path'])
            self.env.close()

    def learn(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5,
              train_timing='on_step_end', train_every_nstep=1, repeat_train=1,
              need_render=True, **kwargs):
        if ctx.enable_tensorboard:
            for k, v in self.training_context.items():
                if isinstance(v, HistoryBase):
                    v.training_name = self.name

            self.training_context['training_name'] = self.name
            self.training_context['summary_writer'] = ctx.summary_writer

            make_dir_if_need(os.path.join(working_directory, 'Logs'))

            t1 = threading.Thread(target=launchTensorBoard, args=([]))
            t1.setDaemon(True)
            t1.start()
            open_browser('http://{0}:{1}/'.format(ctx.tensorboard_server, ctx.tensorboard_port), 5)
        if ctx.enable_mlflow:
            ctx.mlflow_logger.start_run(run_id=self.execution_id)
            open_browser('http://{0}:{1}/'.format(ctx.mlflow_server, ctx.mlflow_port), 5)

        self.training_context['repeat_train'] = repeat_train
        self.training_context['train_timing'] = train_timing
        self.training_context['train_every_nstep'] = train_every_nstep
        self.training_context['min_replay_samples'] = min_replay_samples
        self.training_context['batch_size'] = batch_size

        self.do_on_training_start()

        try:
            self.play(num_episodes=num_episodes, batch_size=batch_size, min_replay_samples=min_replay_samples,
                      print_progess_frequency=print_progess_frequency, training=True,
                      train_timing=train_timing, train_every_nstep=train_every_nstep,
                      repeat_train=repeat_train, need_render=need_render)
        except Exception as e:
            print(e)
            PrintException()

    def resume(self, num_episodes=3000, **kwargs):
        pass

    @property
    def preprocess_flow(self):
        return self._preprocess_flow

    @preprocess_flow.setter
    def preprocess_flow(self, value):
        self._preprocess_flow = value
        objecttype = None
        if self._networks['actor'].signature is not None:
            objecttype = self._networks['actor'].signature.inputs.value_list[0].object_type
        # super()._initial_graph(inputs=to_tensor(self.get_observation()).repeat_elements(2, 0), output=deepcopy(self.network))
        self.setting_network()
        # if objecttype is not None:
        #     self.inputs.value_list[0].object_type = objecttype
        #     self.model.input_spec.object_type = objecttype

        self.env.reset()

    def data_preprocess(self, img_data):
        if is_tensor(img_data):
            img_data = to_numpy(img_data)
        if len(self._preprocess_flow) == 0:
            pass
        else:
            for fn in self.preprocess_flow:
                img_data = fn(img_data)
        img_data = image_backend_adaption(img_data)
        return img_data

    def do_on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start(self.__dict__)
        self.actor.train()
        self.critic.train()

    def do_on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end(self.__dict__)

        self.save_model()
        self.eval()

    def do_on_batch_end(self):
        self.training_context['time_batch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])
        if (self.training_context['steps'] + 1) % ctx.epoch_equivalent == 0:
            if self.warmup > 0 and self.warmup == (self.training_context['steps'] + 1) // ctx.epoch_equivalent:
                self.adjust_learning_rate(self.training_context['base_lr'])
                self.warmup = 0

    def do_on_get_observation_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_get_observation_start(self.training_context)

    def do_on_get_observation_end(self, observation):
        for callback in self.training_context['callbacks']:
            callback.on_get_observation_end(self.training_context)

    def do_on_select_action_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_select_action_start(self.training_context)

    def do_on_select_action_end(self, action):
        for callback in self.training_context['callbacks']:
            callback.on_select_action_end(self.training_context)

    def do_on_get_rewards_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_get_rewards_start(self.training_context)

    def do_on_get_rewards_end(self, observation, reward, done, info):
        for callback in self.training_context['callbacks']:
            callback.on_get_rewards_end(self.training_context)

    def with_tensorboard(self):
        make_dir_if_need(os.path.join(working_directory, 'Logs'))
        # check weather have tensorboard
        if get_backend() == 'pytorch':
            try:
                from trident.loggers.pytorch_tensorboard import SummaryWriter
                ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))

            except Exception as e:
                print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                print(e)
                PrintException()
        elif get_backend() == 'tensorflow':
            try:
                from trident.loggers.tensorflow_tensorboard import SummaryWriter
                ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))

            except Exception as e:
                print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                print(e)
                PrintException()
        return self

    def with_mlflow(self):
        from trident.loggers.mlflow_logger import MLFlowLogger
        ctx.try_enable_mlflow(MLFlowLogger())

        return self


class DqnPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 10000
                 , gamma=0.99, max_epsilon=0.9, min_epsilon=0.01, decay=100
                 , target_update=10, name='dqn') -> None:
        super(DqnPolicy, self).__init__(network=network, env=env, action_strategy=ActionStrategy.OffPolicy, gamma=gamma,
                                        use_experience_replay=True, replay_unit='step',
                                        memory_length=memory_length, name=name)
        self.state = self.env.reset()
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.target_update = target_update
        self.steps_done = 0

    def setting_network(self):
        super()._initial_graph(inputs=self.get_observation(), output=deepcopy(self.network))
        self.policy_net = self._model
        self.policy_net.to(get_device())
        kaiming_normal(self.policy_net)
        self.policy_net.train()

        self.target_net = copy.deepcopy(self.network)
        self.target_net.to(get_device())
        self.target_net.eval()
        self.summary()

    def get_observation(self):

        if hasattr(self.env, 'state'):
            return expand_dims(self.data_preprocess(to_numpy(self.env.state)), 0).astype(np.float32)
        else:
            return expand_dims(self.data_preprocess(self.env.render('observation')), 0).astype(np.float32)

    def select_action(self, state, model_only=False, **kwargs):
        sample = random.random()
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(
            -1.0 * self.steps_done / self.decay)
        self.steps_done += 1
        if model_only == True or sample > self.epsilon:
            with torch.no_grad():
                selected_action = argmax(self.policy_net(to_tensor(state)), 1)
                # self.action_logs['model'][selected_action.item()]+=1
                return selected_action.item()
        else:
            selected_action = np.random.randint(low=0, high=100) % self.env.action_space.n
            # self.action_logs['random'][selected_action] += 1
            return selected_action

    def get_rewards(self, action):
        return self.env.step(action)

    def experience_replay(self, batch_size):
        batch = self.memory.sample(minimum(batch_size, len(self.memory)))
        state_batch = to_tensor(batch.state, requires_grad=True).squeeze(1)
        action_batch = to_tensor(batch.action).long().detach()
        reward_batch = to_tensor(batch.reward).squeeze(1).detach()
        return state_batch, action_batch, batch.next_state, reward_batch

    def calculate_discounted_returns(self, *args, **kwargs):
        state_batch, action_batch, next_state_batch, reward_batch = args

        # 基於目前狀態所產生的獎賞期望值預測 Q(s_t)。
        # 策略網路評估行動
        predict_rewards = self.policy_net(state_batch).gather(1, action_batch)

        target_rewards = zeros(len(next_state_batch))
        # 至於未來狀態的部分，未來狀態主要是計算未來價值使用，但是只要是done，等於不再有未來狀態，則要視成功(1)是失敗(-3)來決定獎賞
        for i in range(len(next_state_batch)):
            s = next_state_batch[i]
            # 要計算未來獎賞預估
            if s is not None:
                # 目標網路評估Q值
                self.target_net.eval()
                next_q = self.target_net(to_tensor(s))
                q_next = max(next_q.detach(), axis=1)

                # 計算目標獎賞值(目前實際獎賞+折扣因子]*未來獎賞預估值)
                target_rewards[i] = reward_batch[i] + (q_next * self.gamma)
            # 不再有未來，只有當期獎賞
            else:
                # 如果是最後一筆表示沒有未來價值
                target_rewards[i] = reward_batch[i] + 0

        # 將計算衍生結果以及data_feed暫存於training_context
        train_data = OrderedDict()
        train_data['state'] = state_batch
        train_data['predict_rewards'] = squeeze(predict_rewards, -1)
        train_data['target_rewards'] = target_rewards.detach()
        train_data['reward_batch'] = reward_batch

        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'predict_rewards'
        data_feed['target'] = 'target_rewards'

        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data

    def save_or_sync_weights(self):
        steps = self.training_context['steps']
        if steps <= 2 * self.accumulation_steps or (steps % self.target_update == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)
            self.save_model(save_path=self.training_context['save_path'])

    def learn(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5,
              train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
              accumulate_grads=False, **kwargs):
        self.play(num_episodes=num_episodes, batch_size=batch_size, min_replay_samples=min_replay_samples,
                  print_progess_frequency=print_progess_frequency, training=True,
                  train_timing=train_timing, train_every_nstep=train_every_nstep,
                  repeat_train=repeat_train, need_render=True)


class PGPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, use_experience_replay=False, memory_length: int = 10000
                 , gamma=0.999, name='pg') -> None:
        super(PGPolicy, self).__init__(network=network, env=env, action_strategy=ActionStrategy.OnPolicy, gamma=gamma,
                                       use_experience_replay=use_experience_replay,
                                       replay_unit='episode',
                                       memory_length=memory_length, name=name)
        self.state = self.env.reset()

        def pg_loss(output, action, rewards):
            loss = 0.0
            for i in range(len(action)):
                probs_i = output[i]
                m = Bernoulli(probs_i)
                action_i = torch.FloatTensor([action[i]]).to(get_device())
                reward_i = rewards[i]
                loss = loss - m.log_prob(action_i) * reward_i
            return loss

        self.with_loss(pg_loss, name='pg_loss')

        self.gamma = gamma

    def setting_network(self):
        super()._initial_graph(inputs=self.get_observation(), output=deepcopy(self.network))
        self.policy_net = self._model

        kaiming_normal(self._model)
        self.policy_net.to(get_device())
        self.summary()
        self.policy_net.train()
        self.steps_done = 0

    def get_observation(self):
        state = None
        if hasattr(self, 'state'):
            state = self.data_preprocess(self.state)
        if hasattr(self.env, 'state'):
            state = self.data_preprocess(self.env.state)
        elif hasattr(self.env, 'screen'):
            state = self.data_preprocess(self.env.screen)
        else:
            state = self.data_preprocess(self.env.render('rgb_array'))
        if ndim(state) == 4:
            return state
        else:
            return np.expand_dims(state, 0)

    def select_action(self, state, **kwargs):
        # 只根據模型來決定action   on-policy
        probs = self.model(state)[0]
        m = Bernoulli(probs)
        action = m.sample()
        return int(action.item())

    def get_rewards(self, action):
        return self.env.step(action)

    def experience_replay(self, batch_size):
        batch = self.memory.sample(builtins.min(batch_size, len(self.memory)))
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        return state_batch, action_batch, None, reward_batch

    def calculate_discounted_returns(self, *args, **kwargs):
        state_pool = []
        action_pool = []
        reward_pool = []
        state_batch, action_batch, _, reward_batch = args
        for i in range(len(state_batch)):
            state_pool.extend(state_batch[i])
            action_pool.extend(action_batch[i])
            reward_pool.extend(reward_batch[i])
        running_add = 0
        # 逆向計算期望獎賞
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == -3:
                running_add = -3
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # 將期望獎賞標準化
        reward_pool = to_numpy(reward_pool).astype(np.float32)
        reward_mean = np.mean(reward_pool.copy())
        reward_std = np.std(reward_pool.copy())
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        train_data = OrderedDict()
        train_data['state'] = to_tensor(np.array(state_pool))
        train_data['output'] = self.policy_net(train_data['state'])
        train_data['action'] = to_tensor(np.array(action_pool)).float()
        train_data['rewards'] = to_tensor(np.array(reward_pool))
        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'output'
        data_feed['action'] = 'action'
        data_feed['rewards'] = 'rewards'

        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data


class A2CPolicy(ActorCriticPolicy):
    """The base class for any RL policy.
    """

    def __init__(self, actor: Layer, critic: Layer, env: gym.Env, gamma=0.999, train_every_nstep=20,
                 name='a2c') -> None:
        super(A2CPolicy, self).__init__(actor=actor, critic=critic, env=env, action_strategy=ActionStrategy.OnPolicy,
                                        gamma=gamma, use_experience_replay=False,
                                        replay_unit='episode', name=name)
        self.state = self.env.reset()
        self.done = False
        self.rollout = Rollout()
        self.train_every_nstep = train_every_nstep
        self.actor.batch_loss_history.regist('actor_loss')
        self.critic.batch_loss_history.regist('critic_loss')

    def get_observation(self):
        self.do_on_get_observation_start()
        state = None
        if hasattr(self, 'state'):
            state = self.data_preprocess(self.state)
        if hasattr(self.env, 'state'):
            state = self.data_preprocess(self.env.state)
        elif hasattr(self.env, 'screen'):
            state = self.data_preprocess(self.env.screen)
        else:
            state = self.data_preprocess(self.env.render('rgb_array'))
        if ndim(state) == 4:
            pass
        else:
            state = np.expand_dims(state, 0)
        self.do_on_get_observation_end(state)
        return state

    def select_action(self, state, **kwargs):
        self.do_on_select_action_start()
        # 只根據模型來決定action   on-policy
        if ndim(state) == 3:
            state = expand_dims(state, 0)
        state = to_tensor(state)
        probs = self.actor.model(state)
        m = Categorical(probs=probs)
        action = m.sample()
        self.rollout.collect('state', state)
        self.rollout.collect('probs', probs)
        self.rollout.collect('action', action)
        self.rollout.collect('log_prob', m.log_prob(action))
        return int(action.item())

    def get_rewards(self, action):
        self.do_on_get_rewards_start()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def calculate_discounted_returns(self, *args, **kwargs):

        if (self.t > 0 and self.t % self.train_every_nstep == 0) or self.done:
            actor_loss = 0
            critic_loss = 0
            reward_pool = self.rollout.get_running_mean('reward')
            log_prob_pool = self.rollout.log_prob
            state_pool = self.rollout.state
            action_pool = self.rollout.action

            running_add = self.critic.model(to_tensor(self.get_observation()))
            # 逆向計算期望獎賞
            for i in reversed(range(len(reward_pool))):
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

            # 將期望獎賞標準化
            reward_pool = stack(reward_pool)
            reward_mean = reward_pool.mean()
            reward_std = reward_pool.std()
            reward_pool = (reward_pool - reward_mean) / reward_std

            for reward, state, log_prob, action in zip(reward_pool, state_pool, log_prob_pool, action_pool):
                qvalues = self.critic.model(to_tensor(state))

                advantage = reward.detach() - qvalues

                critic_loss += advantage ** 2
                actor_loss -= log_prob * (advantage.copy().detach())

            self.critic.current_loss = critic_loss
            self.critic.batch_loss_history.collect('critic_loss', len(self.critic.batch_loss_history),
                                                   critic_loss.copy().detach())
            self.update_critic(self.i_episode, self.t, self.num_episodes, 1, 'on_step_end', self.done,
                               accumulate_grads=False)

            self.actor.current_loss = actor_loss
            self.actor.batch_loss_history.collect('actor_loss', len(self.actor.batch_loss_history),
                                                  actor_loss.copy().detach())
            self.update_actor(self.i_episode, self.t, self.num_episodes, 1, 'on_step_end', self.done,
                              accumulate_grads=False)

            self.rollout.reset()
            # if (self.t > 0 and self.t % (20 * self.train_every_nstep) == 0):
            #     self.actor.print_batch_progress(10 * self.train_every_nstep)
            #     self.critic.print_batch_progress(10 * self.train_every_nstep)
            #     self.save_model()

    def play(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, training=True,
             train_timing='on_episode_end', train_every_nstep=None,
             repeat_train=1,
             need_render=True, **kwargs):
        if train_timing not in ['on_episode_start', 'on_episode_end', 'on_step_end']:
            raise ValueError('Only on_episode_start,on_step_end are valid  train_timing options')
        if train_every_nstep is None:
            train_every_nstep = self.train_every_nstep
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

        self.num_episodes = num_episodes
        self.total_reward = 0
        self.t = 0
        self.i_episode = 0
        if hasattr(self.env, 'recording_enabled'):
            self.env.recording_enabled = True
        for i_episode in range(num_episodes):
            self.total_epoch = num_episodes
            self.current_epoch = i_episode

            self.do_on_epoch_start()
            self.env.reset()
            self.total_rewards = 0
            state = self.get_observation()
            for t in count():
                self.t = t
                self.do_on_batch_start()
                # # Train on_step_start
                # if training and train_timing == 'on_step_start' and t % train_every_nstep == 0:
                #     self.training_model(i_episode, t,num_episodes=num_episodes, repeat_train=repeat_train, batch_size=batch_size)

                action = self.select_action(state, model_only=True)
                self.do_on_select_action_end(action)

                observation, reward, done, info = self.get_rewards(action)
                self.do_on_get_rewards_end(observation, reward, done, info)

                next_state = self.get_observation() if not done else None
                self.done = done
                self.total_rewards += reward
                self.calculate_discounted_returns()

                if need_render:
                    self.env.render()

                complete = self.episode_complete_criteria()
                # Train on_step_end
                if training and train_timing == 'on_episodes_end' and (
                        (self.frame_steps + 1) % train_every_nstep == 0 or done) and self.frame_steps > 5:
                    try:
                        self.training_model(i_episode, t, num_episodes=num_episodes, done=done or complete,
                                            repeat_train=repeat_train, train_timing=train_timing)
                    except Exception as e:
                        print(e)

                state = next_state

                if done or complete:
                    self.epoch_metric_history.collect('rewards', i_episode, float(self.total_rewards))
                    self.epoch_metric_history.collect('t', i_episode, float(t + 1))

                    self.do_on_epoch_end()

                    self.print_epoch_progress()

                    if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                        loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history,
                                          metrics_names=list(self.epoch_metric_history.keys()), calculate_base='epoch',
                                          imshow=True)

                    if self.task_complete_criteria():
                        self.save_model()
                        print('episode {0} meet task complete criteria, training finish! '.format(i_episode))
                        return True

        print('Complete')
        self.env.render()
        self.env.close()


class PPOPolicy(ActorCriticPolicy):
    """Proximal Policy Optimization
    """

    def __init__(self, actor: Layer, critic: Layer, env: gym.Env, state_shape=None, target_update=5, gamma=0.9,
                 beta=0.01, use_gae=False, gae_lambda=0.95, normalize_returns=False, normalize_advantages=False,
                 clip_grad_norm=0.5, name='ppo') -> None:
        super(PPOPolicy, self).__init__(actor=actor, critic=critic, env=env, state_shape=state_shape,
                                        action_strategy=ActionStrategy.OnPolicy, gamma=gamma,
                                        use_experience_replay=False,
                                        replay_unit='episode',
                                        memory_length=10000, name=name)
        self.state = self.env.reset()
        self.target_update = target_update
        self.memory = Rollout()
        self.unsync_cnt = 0

        self.use_gae = use_gae
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.with_grad_clipping(clip_grad_norm)

        self.gamma = gamma

    def setting_network(self):
        if isinstance(self._networks['actor'], Layer):
            self._networks['actor'] = Model(output=self._networks['actor'], inputs=self.get_observation())

        if isinstance(self._networks['critic'], Layer):
            self._networks['critic'] = Model(output=self._networks['critic'], inputs=self.get_observation())
        orthogonal(self._networks['actor'])
        orthogonal(self._networks['critic'])
        self._networks['critic'].training_context['model_name'] = 'critic'
        self._networks['actor'].training_context['model_name'] = 'actor'

        self._networks['critic'].training_context['skip_reset_total_loss'] = True
        self._networks['actor'].training_context['skip_reset_total_loss'] = True
        self._networks['critic'].training_context['skip_generate_output'] = True
        self._networks['actor'].training_context['skip_generate_output'] = True
        self._networks['critic'].training_context['retain_graph'] = True
        self.critic.batch_loss_history.regist('critic_loss')
        self.actor.batch_loss_history.regist('actor_loss')

        self.old_actor = copy.deepcopy(self._networks['actor'].model)
        self.old_actor.eval()
        self.old_actor.trainable = False

    def get_observation(self):
        self.do_on_get_observation_start()
        state = None
        if hasattr(self.env, 'state'):
            state = self.data_preprocess(self.env.state)
        elif hasattr(self.env, 'screen'):
            state = self.data_preprocess(self.env.screen)
        else:
            state = self.data_preprocess(self.env.render('rgb_array'))
        if ndim(state) < 4:
            state = np.expand_dims(state, 0)
        self.state = state
        self.do_on_get_observation_end(state)
        return self.state

    @torch.no_grad()
    def select_action(self, state, **kwargs):
        self.do_on_select_action_start()
        # 只根據模型來決定action   on-policy
        state = to_tensor(state)

        probs = self.old_actor(state).squeeze()
        m = Categorical(probs=probs)
        action = m.sample()

        self.rollout.collect('state', state)
        self.rollout.collect('probs', probs)
        self.rollout.collect('action', action.item())
        self.rollout.collect('log_prob', m.log_prob(action))
        return int(action.item())

    def get_rewards(self, action):
        self.do_on_get_rewards_start()
        observation, reward, done, info = self.env.step(action)
        observation = self.get_observation()

        self.rollout.collect('reward', float(reward))
        self.rollout.collect('done', done)
        self.done = done
        return observation, reward, done, info

    def calculate_discounted_returns(self, *args, **kwargs):
        try:
            actor_loss = 0
            critic_loss = 0
            reward_pool = self.rollout.get_normalized('reward') if self.normalize_returns else self.rollout.reward
            probs_pool = self.rollout.probs
            log_prob_pool = self.rollout.log_prob
            state_pool = self.rollout.state
            action_pool = self.rollout.action
            done_pool = to_tensor(self.rollout.done).squeeze().detach().float()

            last_done = done_pool[-1]

            # running_add = last_next_value
            # DiscountedReturns = []
            Values = []
            Ratio1 = []
            Ratio2 = []
            LogProbs = []
            gae = 0
            R = []

            for state in state_pool:
                value = self.critic.model(state)[0]
                Values.append(value)

            if len(Values) > len(reward_pool):
                Values = Values[:-1]
            Values = torch.cat(Values)

            next_value = to_tensor(0.0)
            if not last_done.bool():
                next_value = self.critic.model(self.state)[0]

            for value, reward, done in list(zip(Values, reward_pool, done_pool))[::-1]:
                gae = gae * self.gamma * self.gae_lambda
                gae = gae + reward + self.gamma * next_value * (1 - done) - value
                next_value = value
                R.append(gae + value)
            R = R[::-1]
            R = torch.cat(R).detach()

            # if self.normalize_returns:
            #     R=(R -R .mean())/(R.std()+1e-5)

            advantages = R - Values

            critic_loss = 0.5 * (advantages ** 2).mean()
            self.critic.training_context['current_loss'] = self.critic.training_context['current_loss'] + critic_loss
            self.critic.training_context['tmp_losses'].collect('critic_loss', self.train_steps,
                                                               critic_loss.copy().detach())
            self.critic.batch_metric_history.collect('critic_rmse', self.train_steps,
                                                     critic_loss.copy().detach().sqrt())

            self.update_critic(self.current_epoch, self.train_steps, self.total_epoch, 1, 'on_step_end', self.done)
            # self.critic.print_batch_progress(1)
            self.critic.current_loss = 0

            # need forward calculate for lstm-based actor
            for i in range(len(reward_pool)):
                new_probs = self.actor(state_pool[i]).squeeze()
                new_m = Categorical(probs=new_probs)
                new_log_prob = new_m.log_prob(to_tensor(action_pool[i]))
                LogProbs.append(new_log_prob)
                ratio1 = exp(new_log_prob - log_prob_pool[i])
                Ratio1.append(ratio1)

            LogProbs = concate(LogProbs, axis=0)
            Ratio1 = concate(Ratio1, axis=0)
            Ratio2 = clip(Ratio1, 1.0 - 0.2, 1.0 + 0.2)

            self.actor.batch_metric_history.collect('ratio1', self.train_steps, Ratio1.mean())
            self.actor.batch_metric_history.collect('ratio2', self.train_steps, Ratio2.mean())

            # surr1 = Ratio1 * advantages
            # surr2 = Ratio2 * advantages
            actor_loss = -1 * (element_min(Ratio1, Ratio2) * advantages).mean()
            return critic_loss, actor_loss, advantages, R, Values
        except Exception as e:
            print(e)
            PrintException()

    def save_or_sync_weights(self):
        self.critic.save_model(save_path=self.critic.training_context['save_path'])
        self.actor.save_model(save_path=self.actor.training_context['save_path'])

        if self.current_epoch <= 2 or (self.done and (self.current_epoch + 1) % self.target_update == 0):
            self.old_actor.load_state_dict(self.actor.model.state_dict())
            self.old_actor.eval()
            self.actor.model.train()
            if len(self.memory) > 3000:
                self.memory.housekeeping(2000)
            print('memory size:{0}'.format(len(self.memory)))
            # self.memory.reset()
        # elif self.train_steps<=2 or (self.unsync_cnt+1) % (self.target_update*self.accumulation_steps)  == 0:
        #     self.old_actor.load_state_dict(self.actor.model.state_dict())
        #     self.old_actor.eval()
        #     self.actor.model.train()
        #     self.unsync_cnt=0
        # else:
        #     self.unsync_cnt+=1

    def training_model(self, current_episode=0, current_step=0, num_episodes=100, train_timing='on_step_end',
                       done=False, repeat_train=1, **kwargs):
        # log_prob_pool = self.rollout.log_prob
        # state_pool = self.rollout.state
        # action_pool = self.rollout.action
        # done_pool = self.rollout.done
        # reward_pool =  self.rollout.get_normalized('reward') if self.normalize_returns else self.rollout.reward
        gae = 0.0
        R = []
        V = []
        V2 = []
        for i in range(repeat_train):
            try:

                actor_loss = to_tensor(0.0)
                critic_loss = to_tensor(0.0)

                if len(self.rollout) > 3:
                    for module in self.actor.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = True
                                module.clear_state()
                    for module in self.critic.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = True
                                module.clear_state()

                    accumulate_grads = (self.training_context['steps'] + 1) % self.accumulation_steps != 0
                    try:
                        self.train_steps += 1
                        self.current_batch += 1
                        self.steps = self.train_steps
                        self.actor.steps = self.train_steps
                        self.critic.steps = self.train_steps
                        self.done = done

                        critic_loss, actor_loss, advantages, R, V = self.calculate_discounted_returns()

                        # keep = []
                        # for t in list(range(len(state_pool)))[::-1]:
                        #     if done_pool[t] == True or reward_pool[t] > 0.15 or reward_pool[t] < -0.15:
                        #         keep.extend([t, t - 1, t - 2, t - 3, t - 4])
                        #     elif random.random() < 0.05:
                        #         keep.extend([t, t - 1])
                        #
                        # def not_negative(n):
                        #     return n >= 0
                        #
                        # keep = list(filter(not_negative, list(sorted((set(keep))))))
                        # for idx in keep:

                        self.actor.training_context['current_loss'] = self.actor.training_context[
                                                                          'current_loss'] + actor_loss
                        self.actor.training_context['tmp_losses'].collect('actor_loss', self.train_steps,
                                                                          actor_loss.copy().detach())

                        # self.actor.batch_loss_history.collect('entropy_loss', self.train_steps, (-0.01 * entropy_loss).copy().detach() / (len(state_pool) - 1))

                        self.update_actor(self.current_epoch, self.train_steps, self.total_epoch, 1, 'on_step_end',
                                          self.done, accumulate_grads=accumulate_grads)
                        self.actor.current_loss = 0.0
                        # self.actor.print_batch_progress(1)
                        # self.save_or_sync_weights()

                        for idx in list(range(len(R)))[::-1]:
                            if random.random() > 0.9 or abs(self.rollout.reward[idx]) >= 0.12:
                                self.memory.collect('return', R[idx])
                                self.memory.collect('state', to_numpy(self.rollout.state[idx]))
                                self.memory.collect('action', self.rollout.action[idx])
                                self.memory.collect('log_prob', self.rollout.log_prob[idx])

                        self.rollout.reset()
                        del advantages
                        del V
                        gc.collect()

                    except Exception as e:
                        print(e)
                        PrintException()

                if self.done == True and len(self.memory) > 5:
                    for module in self.actor.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = False
                    for module in self.critic.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = False

                    for i in range(8):
                        try:
                            self.train_steps += 1
                            self.current_batch += 1
                            self.steps = self.train_steps
                            accumulate_grads = (self.training_context['steps'] + 1) % self.accumulation_steps != 0
                            # self.do_on_batch_start()

                            batch_data = self.memory.get_samples(
                                builtins.min(self.training_context['batch_size'], len(self.memory)))
                            batch_return = batch_data['return']
                            batch_state = batch_data['state']
                            batch_action = batch_data['action']
                            batch_log_prob = batch_data['log_prob']

                            batch_value = concate(
                                [self.critic(to_tensor(batch_state[i]))[0] for i in range(len(batch_state))], axis=0)
                            batch_return = to_tensor(batch_return)
                            batch_advantages = (batch_return.detach() - batch_value)

                            batch_critic_loss = 0.5 * (batch_advantages ** 2).mean()
                            self.critic.training_context['current_loss'] = self.critic.training_context[
                                                                               'current_loss'] + batch_critic_loss
                            self.critic.training_context['tmp_losses'].collect('critic_loss', self.train_steps,
                                                                               batch_critic_loss.copy().detach())

                            self.critic.batch_metric_history.collect('critic_rmse', self.train_steps,
                                                                     (batch_critic_loss.copy().detach()).sqrt())
                            self.update_critic(self.current_epoch, self.train_steps, self.total_epoch, 1, 'on_step_end',
                                               False, accumulate_grads=accumulate_grads)
                            self.critic.current_loss = 0.0
                            batch_actor_loss = to_tensor(0.0)
                            for j in range(len(batch_state)):
                                # old_batch_probs = self.old_actor(to_tensor(batch_state[j]))
                                # old_m = Categorical(probs=old_batch_probs)
                                old_batch_probs = batch_log_prob[j].detach()
                                # old_batch_probs=old_m.log_prob(to_tensor(batch_action[j]).long())
                                new_batch_probs = self.actor(to_tensor(batch_state[j]))
                                new_m = Categorical(probs=new_batch_probs)
                                new_log_prob = new_m.log_prob(to_tensor(batch_action[j]).long().detach())

                                # entropy_loss = entropy_loss + new_m.entropy().mean()
                                ratio = exp(new_log_prob - old_batch_probs)
                                ratio2 = clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
                                # surr1 = ratio * (batch_advantages[j].detach())
                                # surr2 = clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * (batch_advantages[j].detach())
                                batch_actor_loss = batch_actor_loss - (
                                            torch.min(ratio, ratio2) * batch_advantages[j]).mean()

                            self.actor.training_context['current_loss'] = self.actor.training_context[
                                                                              'current_loss'] + batch_actor_loss / len(
                                batch_state)
                            self.actor.training_context['tmp_losses'].collect('actor_loss', self.train_steps, (
                                        batch_actor_loss / len(batch_state)).copy().detach())

                            # self.actor.batch_loss_history.collect('entropy_loss', self.train_steps, (-0.01 * entropy_loss).copy().detach() / (len(state_pool) - 1))

                            self.update_actor(self.current_epoch, self.train_steps, self.total_epoch, 1, 'on_step_end',
                                              False, accumulate_grads=accumulate_grads)
                            self.actor.current_loss = 0.0
                        # self.save_or_sync_weights()
                        except Exception as e:
                            print(e)
                            PrintException()
                    for module in self.actor.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = True
                    for module in self.critic.model.modules():
                        class_name = module.__class__.__name__.lower()
                        if 'lstm' in class_name or 'gru' in class_name:
                            if hasattr(module, 'stateful'):
                                module.stateful = True

                if self.done == True:
                    print(gray_color(
                        'episode {0}  t: {1} total rewards:{2:.2f} {3} '.format(self.current_epoch + 1, self.t,
                                                                                self.total_rewards, ''.join(
                                ['{0}: {1} '.format(k, v) for k, v in self.info.items() if
                                 k not in ['world', 'y_pos', 'x_pos']]))))
                    print('')
                    self.save_or_sync_weights()

            except Exception as e:
                print(e)
                PrintException()

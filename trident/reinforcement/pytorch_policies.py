import copy
import math
import random
import time
import uuid
import builtins
from copy import deepcopy
from itertools import count
from types import MethodType

import numpy as np
import torch
from gym.spaces import Box
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from trident.backend.common import *
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_ops import *
from trident.backend.tensorspec import TensorSpec, ObjectType
from trident.data.image_common import image_backend_adaption
from trident.misc.visualization_utils import loss_metric_curve
from trident.layers.pytorch_initializers import kaiming_normal
from trident.optims.pytorch_trainer import Model
from trident.reinforcement.utils import ReplayBuffer, ActionStrategy, ObservationType

import_or_install('gym')
import gym

_session = get_session()
__all__ = ['PolicyBase', 'DqnPolicy', 'A2CPolicy']


def modify_env(env: gym.Env):
    def render(env: gym.Env, mode='human'):
        _env = None
        if isinstance(env, gym.core.Wrapper):
            _env = env.env
        elif isinstance(env, gym.core.Env):
            _env = env

        if mode == 'human':
            return _env.render('human')
        elif mode == 'observation':
            if env.obsetvation_type == ObservationType.Image and hasattr(env, 'observation') and callable(env.observation):
                return env.observation(_env.render('rgb_array').copy())
            else:
                return _env.render('rgb_array')
        elif mode == 'rgb_array':
            return _env.render('rgb_array')

    if isinstance(env.observation_space, Box) and len(env.observation_space.shape) in [2, 3] and (
            (env.observation_space.low.max() == 0 and env.observation_space.high.min() == 255)
            or env.observation_space.dtype == np.uint8):
        env.obsetvation_type = ObservationType.Image
    elif isinstance(env.observation_space, Box) and env.observation_space.dtype == np.float32:
        env.obsetvation_type = ObservationType.Box

    setattr(env, 'render', MethodType(render, env))

    return env


class PolicyBase(Model):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, action_strategy=None, gamma=0.99, use_experience_replay=False, replay_unit='step', memory_length: int = 10000,
                 name=None) -> None:
        super().__init__()
        self.network = network
        if name is not None:
            self.network._name = name

        self.env = modify_env(env)
        self.env.reset()
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
        self.state_pool = []
        self.reward_pool = []
        self.action_pool = []
        self.value_pool = []
        self.setting_network()

    def setting_network(self):
        super()._initial_graph(inputs=self.get_observation(), output=deepcopy(self.network))

    def get_observation(self):
        if hasattr(self.env, 'state'):
            return expand_dims(self.data_preprocess(self.env.state), 0)
        else:
            return expand_dims(self.data_preprocess(self.env.render('observation')), 0)

    def select_action(self, state, model_only=False, **kwargs):
        return self.env.action_space.samples()

    def get_rewards(self, action):
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
                action = self.select_action(state, model_only=True if self.action_strategy == ActionStrategy.OnPolicy else False)
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
                        #                                                                                       float(n4) / (n3 + n4), float(n3 + n4) / builtins.max(n1 + n2 + n3 + n4,1)))
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

    def estimate_future_return(self, *args, **kwargs):
        return NotImplemented

    def save_or_sync_weights(self):
        self.save_model(save_path=self.training_context['save_path'])

    def training_model(self, current_episode=0, current_step=0, num_episodes=100, train_timing=None, done=False, batch_size=1, repeat_train=1):

        is_collect_data = False
        for i in range(repeat_train):
            data = None
            if self.use_experience_replay:
                data = self.experience_replay(batch_size)
            else:
                data = self.memory.memory[0]
            self.estimate_future_return(*data)
            self.training_context['skip_generate_output'] = True
            if 'step' in train_timing:
                current_step = current_step * repeat_train + i
                if done:
                    total_batch = current_step * repeat_train + i + 1
                    is_collect_data = True
                else:
                    total_batch = current_step * repeat_train + i + 10
            elif 'episode' in train_timing:
                current_step = i
                total_batch = repeat_train

            super(PolicyBase, self).train_model(self.training_context['train_data'], self.training_context['test_data'],
                                                current_epoch=current_episode,
                                                current_batch=current_step,
                                                total_epoch=num_episodes,
                                                total_batch=total_batch,
                                                is_collect_data=True,
                                                is_print_batch_progress=False,
                                                is_print_epoch_progress=False,
                                                log_gradients=False, log_weights=False,
                                                accumulate_grads=(current_step * repeat_train + 1) % self.accumulation_steps != 0)
            self.save_or_sync_weights()

    def play(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, training=True, train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
             need_render=True):
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
        self.state_pool = []
        self.reward_pool = []
        self.action_pool = []

        self.total_reward = 0
        self.t = 0
        self.i_episode = 0
        if hasattr(self.env, 'recording_enabled'):
            self.env.recording_enabled = True
        for i_episode in range(num_episodes):
            self.i_episode = i_episode

            if training and train_timing == 'on_episode_start' and i_episode % train_every_nstep == 0:
                self.training_model(i_episode, 0, num_episodes=num_episodes, repeat_train=repeat_train, train_timing=train_timing, batch_size=batch_size)
            self.env.reset()
            self.total_rewards = 0
            state = self.get_observation()
            for t in count():
                self.t = t
                # # Train on_step_start
                # if training and train_timing == 'on_step_start' and t % train_every_nstep == 0:
                #     self.training_model(i_episode, t,num_episodes=num_episodes, repeat_train=repeat_train, batch_size=batch_size)

                action = self.select_action(state, model_only=True)
                observation, reward, done, info = self.get_rewards(action)

                self.total_rewards += reward

                next_state = self.get_observation() if not done else None

                if need_render:
                    self.env.render()
                if self.replay_unit == 'step':
                    if self.push_into_memory_criteria(state, action, next_state, reward) or done:
                        self.memory.push(state, action, next_state, reward)
                elif self.replay_unit == 'episode':
                    self.state_pool.append(state)
                    self.action_pool.append(action)
                    self.reward_pool.append(reward)
                    if done:
                        if self.push_into_memory_criteria(self.state_pool, self.action_pool, None, self.reward_pool):
                            self.memory.push(self.state_pool, self.action_pool, None, self.reward_pool)
                        self.state_pool = []
                        self.action_pool = []
                        self.reward_pool = []

                complete = self.episode_complete_criteria()
                # Train on_step_end
                if training and train_timing == 'on_step_end' and t % train_every_nstep == 0:
                    self.training_model(i_episode, t, num_episodes=num_episodes, done=done or complete, repeat_train=repeat_train, train_timing=train_timing, batch_size=batch_size,
                                        accumulate_grads=accumulate_grads)

                state = next_state
                if done or complete:
                    self.epoch_metric_history.collect('rewards', i_episode, float(self.total_rewards))
                    self.epoch_metric_history.collect('t', i_episode, float(t + 1))
                    if self.use_experience_replay:
                        self.epoch_metric_history.collect('replay_buffer_utility', i_episode, float(len(self.memory)) / self.memory.capacity)

                    if print_progess_frequency == 1 or (i_episode > 0 and (i_episode + 1) % print_progess_frequency == 0):
                        self.print_epoch_progress(print_progess_frequency)
                        # n1 = self.action_logs['model'][0]
                        # n2 = self.action_logs['model'][1]
                        # n3 = self.action_logs['random'][0]
                        # n4 = self.action_logs['random'][1]
                        # print('model: 0:{0} 1:{1}  random: 0:{2} 1:{3}  random: {4}'.format(float(n1) / (n1 + n2), float(n2) / (n1 + n2), float(n3) / builtins.max(n3 + n4,1),
                        #                                                                                       float(n4) / builtins.max(n3 + n4,1), float(n3 + n4) / builtins.max(n1 + n2 + n3 + n4,1)))
                        #
                        # self.action_logs = OrderedDict()
                        # self.action_logs['model'] = OrderedDict()
                        # self.action_logs['random'] = OrderedDict()
                        # self.action_logs['model'][0] = 0
                        # self.action_logs['model'][1] = 0
                        # self.action_logs['random'][0] = 0
                        # self.action_logs['random'][1] = 0
                    # 定期繪製損失函數以及評估函數對時間的趨勢圖
                    if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                        loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history, metrics_names=list(self.epoch_metric_history.keys()), calculate_base='epoch',
                                          imshow=True)

                    if self.task_complete_criteria():
                        self.save_model(save_path=self.training_context['save_path'])
                        print('episode {0} meet task complete criteria, training finish! '.format(i_episode))
                        return True

                    break

        print('Complete')
        self.env.render()
        self.env.close()

    def learn(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
              accumulate_grads=False):
        self.play(num_episodes=num_episodes, batch_size=batch_size, min_replay_samples=min_replay_samples, print_progess_frequency=print_progess_frequency, training=True,
                  train_timing=train_timing, train_every_nstep=train_every_nstep,
                  repeat_train=repeat_train, need_render=True)

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
                if self._model is not None and self.signature is not None and len(self.signature) > 1 and self._model.input_spec is not None:
                    img_data = fc(img_data, spec=self._model.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)
            if self._model.input_spec is None:
                self._model.input_spec = TensorSpec(shape=tensor_to_shape(to_tensor(img_data), need_exclude_batch_axis=True, is_singleton=True), object_type=ObjectType.rgb,
                                                    name='input')

                self.input_shape = self._model.input_spec.shape[1:]

            return img_data
        else:
            return img_data

    def do_on_batch_end(self):
        self.training_context['time_batch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['steps'] += 1
        if (self.training_context['steps'] + 1) % _session.epoch_equivalent == 0:
            if self.warmup > 0 and self.warmup == (self.training_context['steps'] + 1) // _session.epoch_equivalent:
                self.adjust_learning_rate(self.training_context['base_lr'])
                self.warmup = 0


class DqnPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 10000
                 , gamma=0.99, max_epsilon=0.9, min_epsilon=0.01, decay=100
                 , target_update=10, name='dqn') -> None:
        super(DqnPolicy, self).__init__(network=network, env=env, action_strategy=ActionStrategy.OffPolicy, gamma=gamma, use_experience_replay=True, replay_unit='step',
                                        memory_length=memory_length, name=name)

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
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1.0 * self.steps_done / self.decay)
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

    def estimate_future_return(self, *args, **kwargs):
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

    def learn(self, num_episodes, batch_size=1, min_replay_samples=1, print_progess_frequency=5, train_timing='on_episode_start', train_every_nstep=1, repeat_train=1,
              accumulate_grads=False):
        self.play(num_episodes=num_episodes, batch_size=batch_size, min_replay_samples=min_replay_samples, print_progess_frequency=print_progess_frequency, training=True,
                  train_timing=train_timing, train_every_nstep=train_every_nstep,
                  repeat_train=repeat_train, need_render=True)


class PGPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, use_experience_replay=False, memory_length: int = 10000
                 , gamma=0.999, name='pg') -> None:
        super(PGPolicy, self).__init__(network=network, env=env, action_strategy=ActionStrategy.OnPolicy, gamma=gamma, use_experience_replay=use_experience_replay,
                                       replay_unit='episode',
                                       memory_length=memory_length, name=name)

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
        if hasattr(self.env, 'state'):
            return expand_dims(self.data_preprocess(to_numpy(self.env.state)), 0).astype(np.float32)
        else:
            return expand_dims(self.data_preprocess(self.env.render('observation')), 0).astype(np.float32)

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

    def estimate_future_return(self, *args, **kwargs):
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


class A2CPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, use_experience_replay=False, memory_length: int = 10000
                 , gamma=0.999, name='a2c') -> None:
        super(A2CPolicy, self).__init__(network=network, env=env, action_strategy=ActionStrategy.OnPolicy, gamma=gamma, use_experience_replay=use_experience_replay,
                                        replay_unit='episode',
                                        memory_length=memory_length, name=name)

        def policy_loss(policy_losses):
            return policy_losses.sum()

        def value_loss(value_losses):
            return value_losses.sum()

        self.with_loss(policy_loss, name='policy_loss')
        self.with_loss(value_loss, name='value_loss')

        self.gamma = gamma

    def setting_network(self):
        super()._initial_graph(inputs=self.get_observation(), output=deepcopy(self.network))
        self.actor_critic = self._model

        kaiming_normal(self._model)
        self.actor_critic.to(get_device())
        self.summary()
        self.actor_critic.train()
        self.steps_done = 0

    def get_observation(self):
        return expand_dims(self.data_preprocess(self.env.render('observation')), 0).astype(np.float32)

    def select_action(self, state, **kwargs):
        # 只根據模型來決定action   on-policy
        if ndim(state) == 3:
            state = expand_dims(state, 0)
        state = to_tensor(state)
        out = self.model(state)
        if isinstance(out, OrderedDict):
            out = out.value_list
        probs, qvalue = out
        try:
            probs = where(is_abnormal_number(probs), zeros_like(probs), probs)
            m = Categorical(probs=probs)
            action = m.sample()
            return int(action.item())
        except Exception  as e:
            print(probs)
            print(e)
            action = argmax(probs, -1)
            return int(action.item())

    def get_rewards(self, action):
        return self.env.step(action)

    def experience_replay(self, batch_size):
        batch = self.memory.sample(builtins.min(batch_size, len(self.memory)))
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        return state_batch, action_batch, batch.next_state, reward_batch

    def estimate_future_return(self, *args, **kwargs):
        policy_losses = []
        value_losses = []
        state_batch, action_batch, _, reward_batch = args
        reward_batch = reward_batch.astype(np.float32)
        running_add = 0.0
        # 逆向計算期望獎賞
        for i in reversed(range(len(reward_batch))):
            if i == len(reward_batch) - 1:
                running_add = reward_batch[i]
            else:
                running_add = running_add * self.gamma + reward_batch[i]
                reward_batch[i] = running_add
        # 將期望獎賞標準化
        reward_batch = np.array(reward_batch, dtype=np.float32)
        reward_mean = np.mean(reward_batch.copy())
        reward_std = np.std(reward_batch.copy())
        reward_batch = np.expand_dims(((reward_batch - reward_mean) / reward_std), -1)
        policy_losses = to_tensor(0.0)
        value_losses = to_tensor(0.0)
        for action, state, expect_reward in zip(action_batch, state_batch, reward_batch):
            state = to_tensor(state)
            out = self.model(state)
            action_log_prob = None
            if isinstance(out, OrderedDict):
                out = out.value_list
            probs, value = out
            try:
                probs = where(is_abnormal_number(probs), zeros_like(probs), probs)
                m = Categorical(probs=probs)
                action_log_prob = m.log_prob(to_tensor(action))

            except Exception  as e:
                print(probs)
                print(e)

            expect_reward = to_tensor(expect_reward)
            advantage = (expect_reward - value).detach()

            # calculate actor (policy) loss
            policy_losses = policy_losses + (-action_log_prob * (advantage.detach()))

            # calculate critic (value) loss using L1 smooth loss
            value_losses = value_losses + ((value - expect_reward) ** 2)

        train_data = OrderedDict()
        train_data['policy_losses'] = true_divide(policy_losses, builtins.max(len(reward_batch), 1))
        train_data['value_losses'] = true_divide(value_losses, builtins.max(len(reward_batch), 1))
        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'output'
        data_feed['value_losses'] = 'value_losses'
        data_feed['policy_losses'] = 'policy_losses'

        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data
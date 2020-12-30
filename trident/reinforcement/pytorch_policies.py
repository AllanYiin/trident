import copy
from copy import deepcopy
import math
import random
from itertools import count
import torch
import numpy as np
from matplotlib.pylab import plt
from torch import nn
from typing import Any, List, Union, Mapping, Optional, Callable

from trident.backend.tensorspec import TensorSpec, ObjectType
from trident.misc.visualization_utils import loss_metric_curve

from trident.data.image_common import image_backend_adaption
from trident.backend.common import *
from trident.backend.pytorch_ops import *
from trident.backend.pytorch_backend import *
from trident.optims.pytorch_optimizers import Optimizer, get_optimizer
from trident.optims.pytorch_trainer import Model
from trident.optims.pytorch_losses import *
from trident.reinforcement.utils import ReplayBuffer, Transition

import_or_install('gym')
import gym

__all__ = ['PolicyBase', 'DqnPolicy']


class PolicyBase(Model):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 1000, name=None) -> None:
        super().__init__()
        self.network = network
        if name is not None:
            self.network._name=name
        self.env = env
        self.env.reset()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
       # self.agent_id = self.uuid
        self.memory = ReplayBuffer(memory_length)
        self.name = name
        self.setting_network()

    def setting_network(self):
        super()._initial_graph(input_shape=int_shape(self.get_observation())[1:],output=deepcopy(self.network))

    def get_observation(self):
        return self.data_preprocess(self.env.render('rgb_array'))

    def select_action(self, state, **kwargs):
        pass

    def get_rewards(self, action):
        observation_, reward, done, info = self.env.step(action.item())
        return reward

    def experience_replay(self, batch_size):
        train_data = OrderedDict()

        return train_data


    def learn(self, num_episodes=3000, **kwargs):
        pass

    def resume(self, num_episodes=3000, **kwargs):
        pass

    @property
    def preprocess_flow(self):
        return self._preprocess_flow

    @preprocess_flow.setter
    def preprocess_flow(self, value):
        self._preprocess_flow = value
        if isinstance(self.input_spec, TensorSpec):
            self.input_spec = None
        super()._initial_graph(inputs=to_tensor(self.get_observation()).repeat_elements(2, 0), output=deepcopy(self.network))
        self.setting_network()

        self.env.reset()
    def data_preprocess(self, img_data):
        if not hasattr(self,'_preprocess_flow') or self._preprocess_flow is None:
            self._preprocess_flow=[]
        if img_data.ndim==4:
            return to_tensor(to_numpy([self.data_preprocess(im) for im in img_data]))
        if len(self._preprocess_flow) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self._preprocess_flow:
                if self._model is not None and self.signature is not None and len(self.signature) > 1 and self.input_spec is not None:
                    img_data = fc(img_data,spec=self.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)
            if self.input_spec is None :
                self._model.input_spec= TensorSpec(shape=tensor_to_shape(to_tensor(img_data),need_exclude_batch_axis=False), object_type=ObjectType.rgb, name='input')

                self.input_shape=self._model.input_spec.shape[1:]

            return img_data
        else:
            return img_data



class Dqn(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 100000, gamma=0.9, max_epsilon=0.9, min_epsilon=0.01, decay=200, target_update=10, batch_size=10,
                 name='dqn') -> None:
        super().__init__(network=network, env=env, memory_length=memory_length, name=name)

        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.steps_done = 0

    def setting_network(self):
        super()._initial_graph(inputs=to_tensor(self.get_observation()).repeat_elements(2, 0), output=copy.deepcopy(self.network))

        self.policy_net = self.model
        self.policy_net.train()

        self.target_net = deepcopy(self.network)
        self.target_net.eval()
        self.summary()

    def get_observation(self):
        # 在這邊維護取得STATE的方法
        return np.expand_dims(np.array(list(self.env.state)), 0)

    def select_action(self, state, **kwargs):
        # 在這邊維護智能體如何選擇行動的邏輯
        # max_epsilon = 0.9  #初期還沒有案例可以供建模，因此大部分根據隨機案例
        # min_epsilon = 0.01 # 即使模型準確率越來越高，還是必須保留部分比例基於隨機案例
        # decay =200   # 衰減速度

        sample = random.random()
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1.0 * self.steps_done / self.decay)

        self.steps_done += 1
        if sample > self.epsilon:
            self.policy_net.eval()
            selected_action = argmax(self.policy_net(to_tensor(state))[0])
            return selected_action.item()
        else:
            selected_action = np.random.randint(low=0, high=self.env.action_space.n)
            return selected_action

    def get_rewards(self, action):
        # Define the method how to get rewards
        # step to next time.
        observation_, reward, done, info = self.env.step(action)
        return reward if not done else -10 * done, done

    def experience_replay(self, batch_size):
        # Experimenttal Replay

        batch = self.memory.sample(batch_size)

        next_state_batch = to_tensor(batch.next_state, requires_grad=True).squeeze(1)
        state_batch = to_tensor(batch.state, requires_grad=True).squeeze(1)
        action_batch = to_tensor(batch.action).long().detach()
        reward_batch = to_tensor(batch.reward).squeeze(1).detach()

        # Predict expected rewards Q(s_t) base on current state。
        self.policy_net.eval()
        predict_rewards = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Predict future rewards Q(s_{t+1}) base on next state。
        next_q = self.target_net(next_state_batch)
        q_next = max(next_q, axis=-1)

        # Calculate target rewards base on  Bellmann-equation.
        # 𝑄(𝑠,𝑎)=𝑟0+𝛾max𝑎𝑄∗(𝑠′,𝑎)
        target_rewards = reward_batch + (q_next * self.gamma) * greater(reward_batch, 0)
        target_rewards = target_rewards.detach()

        train_data = OrderedDict()
        train_data['state'] = state_batch
        train_data['predict_rewards'] = predict_rewards
        train_data['target_rewards'] = target_rewards
        train_data['reward_batch'] = reward_batch
        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'predict_rewards'
        data_feed['target'] = 'target_rewards'
        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data
        return train_data

    def learn(self, num_episodes=300, batch_size=None, print_progess_frequency=10, imshow=True):
        """The main method for the agent learn

        Returns:
            object:
        """
        if batch_size is not None:
            self.batch_size = batch_size

        self.steps_done = 0
        for i_episode in range(num_episodes):
            # reset enviorment
            self.env.reset()
            # clear rewards
            total_rewards = 0
            state = self.get_observation()

            # 需要記憶中的案例數大於批次數才開始訓練
            start_train = (len(self.memory) > self.batch_size)
            for t in count():
                # 基於目前狀態產生行動
                action = self.select_action(state)
                # 基於行動產生獎賞以及判斷是否結束(此時已經更新至下一個時間點)
                reward, done = self.get_rewards(action)
                # 累積獎賞
                total_rewards += reward

                # 任務完成強制終止(以300為基礎)
                conplete = (not done and t + 1 >= 300)

                if imshow:
                    # 更新視覺化螢幕
                    self.env.render()
                # get next state
                next_state = self.get_observation()

                # 將四元組儲存於記憶中，建議要減少「好案例」的儲存比例
                #if reward<1.0 or (reward>=1.0 and i_episode<50 ) or (reward>=1.0 and i_episode>=50 and random.random()<0.5):
                self.memory.push(state, action, next_state, reward)

                # switch next t
                state =next_state

                if start_train:
                    # get batch data from experimental replay
                    trainData = self.experience_replay(self.batch_size)
                    # switch model to training mode
                    self.policy_net.train()
                    self.train_model(trainData, None,
                                     current_epoch=i_episode,
                                     current_batch=t,
                                     total_epoch=num_episodes,
                                     total_batch=t + 1 if done or conplete else t + 2,
                                     is_collect_data=True if done or conplete else False,
                                     is_print_batch_progress=False,
                                     is_print_epoch_progress=False,
                                     log_gradients=False, log_weights=False,
                                     accumulate_grads=False)

                if done or conplete:
                    if start_train:

                        # self.epoch_metric_history.collect('episode_durations',i_episode,float(t))
                        # 紀錄累積獎賞
                        self.epoch_metric_history.collect('total_rewards', i_episode, float(total_rewards))
                        # 紀錄完成比率(以200為基礎)
                        self.epoch_metric_history.collect('task_complete', i_episode, 1.0 if t + 1 >= 200 else 0.0)
                        # 定期列印學習進度
                        if i_episode % print_progess_frequency == 0:
                            self.print_epoch_progress(print_progess_frequency)
                        # 定期繪製損失函數以及評估函數對時間的趨勢圖
                        if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                            print('epsilon:', self.epsilon)
                            print('predict_rewards:', self.training_context['train_data']['predict_rewards'][:5])
                            print('target_rewards:', self.training_context['train_data']['target_rewards'][:5])
                            print('reward_batch:', self.training_context['train_data']['reward_batch'][:5])
                            loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history,legend=['dqn'], calculate_base='epoch', imshow=imshow)

                    break

            # 定期更新target_net權值
            if start_train and i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)
                self.save_model(save_path=self.training_context['save_path'])

        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()


DqnPolicy = Dqn


class PolicyGradient(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 100000, gamma=0.9, max_epsilon=0.9, min_epsilon=0.01, decay=200, target_update=10, batch_size=10,
                 name='pg') -> None:
        super(PolicyGradient, self).__init__(network=network, env=env, memory_length=memory_length, name=name)


        self.target_net = deepcopy(self._model)
        self.target_net.eval()
        self.summary()

        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.steps_done = 0

    def get_observation(self):
        # 在這邊維護取得STATE的方法
        return np.expand_dims(np.array(list(self.env.state)), 0)

    def select_action(self, state, **kwargs):

        sample = random.random()
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1.0 * self.steps_done / self.decay)

        self.steps_done += 1
        if sample > self.epsilon:
            self.policy_net.eval()
            selected_action = argmax(self.policy_net(to_tensor(state))[0])
            return selected_action.item()
        else:
            selected_action = np.random.randint(low=0, high=self.env.action_space.n)
            return selected_action

    def discount_rewards(self, r: Tensor, gamma: float = 0.999):
        """使用1D rewards向量以及計算折價後獎賞 """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def get_rewards(self, action):
        # Define the method how to get rewards
        # step to next time.
        observation_, reward, done, info = self.env.step(action)
        return reward if not done else -10 * done, done

    def experience_replay(self, batch_size):
        # Experimenttal Replay

        batch = self.memory.sample(batch_size)

        next_state_batch = to_tensor(batch.next_state, requires_grad=True).squeeze(1)
        state_batch = to_tensor(batch.state, requires_grad=True).squeeze(1)
        action_batch = to_tensor(batch.action).long().detach()
        reward_batch = to_tensor(batch.reward).squeeze(1).detach()

        # Predict expected rewards Q(s_t) base on current state。
        self.policy_net.eval()
        predict_rewards = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Predict future rewards Q(s_{t+1}) base on next state。
        next_q = self.target_net(next_state_batch)
        q_next = max(next_q, axis=-1)

        # Calculate target rewards base on  Bellmann-equation.
        # 𝑄(𝑠,𝑎)=𝑟0+𝛾max𝑎𝑄∗(𝑠′,𝑎)
        target_rewards = reward_batch + (q_next * self.gamma) * greater(reward_batch, 0)
        target_rewards = target_rewards.detach()

        train_data = OrderedDict()
        train_data['state'] = state_batch
        train_data['predict_rewards'] = predict_rewards
        train_data['target_rewards'] = target_rewards
        train_data['reward_batch'] = reward_batch
        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'predict_rewards'
        data_feed['target'] = 'target_rewards'
        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data
        return train_data

    def learn(self, num_episodes=300, batch_size=None, print_progess_frequency=10, imshow=True):
        """The main method for the agent learn

        Returns:
            object:
        """
        if batch_size is not None:
            self.batch_size = batch_size

        self.steps_done = 0
        for i_episode in range(num_episodes):
            # reset enviorment
            self.env.reset()
            # clear rewards
            total_rewards = 0
            state = self.get_observation()

            # 需要記憶中的案例數大於批次數才開始訓練
            start_train = (len(self.memory) > self.batch_size)
            for t in count():
                # 基於目前狀態產生行動
                action = self.select_action(state)
                # 基於行動產生獎賞以及判斷是否結束(此時已經更新至下一個時間點)
                reward, done = self.get_rewards(action)
                # 累積獎賞
                total_rewards += reward

                # 任務完成強制終止(以300為基礎)
                conplete = (not done and t + 1 >= 300)

                if imshow:
                    # 更新視覺化螢幕
                    self.env.render()
                # get next state
                next_state = self.get_observation()

                # 將四元組儲存於記憶中，建議要減少「好案例」的儲存比例
                if reward < 1 or (reward == 1 and i_episode < 20) or (
                        reward == 1 and i_episode >= 20 and t < 100 and random.random() < 0.1 and i_episode >= 20 and t >= 100 and random.random() < 0.2):
                    self.memory.push(state, action, next_state, reward)

                # switch next t
                state = deepcopy(next_state)

                if start_train:
                    # get batch data from experimental replay
                    trainData = self.experience_replay(self.batch_size)
                    # switch model to training mode
                    self.policy_net.train()
                    self.train_model(trainData, None,
                                     current_epoch=i_episode,
                                     current_batch=t,
                                     total_epoch=num_episodes,
                                     total_batch=t + 1 if done or conplete else t + 2,
                                     is_collect_data=True if done or conplete else False,
                                     is_print_batch_progress=False,
                                     is_print_epoch_progress=False,
                                     log_gradients=False, log_weights=False,
                                     accumulate_grads=False)

                if done or conplete:
                    if start_train:

                        # self.epoch_metric_history.collect('episode_durations',i_episode,float(t))
                        # 紀錄累積獎賞
                        self.epoch_metric_history.collect('total_rewards', i_episode, float(total_rewards))
                        # 紀錄完成比率(以200為基礎)
                        self.epoch_metric_history.collect('task_complete', i_episode, 1.0 if t + 1 >= 200 else 0.0)
                        # 定期列印學習進度
                        if i_episode % print_progess_frequency == 0:
                            self.print_epoch_progress(print_progess_frequency)
                        # 定期繪製損失函數以及評估函數對時間的趨勢圖
                        if i_episode > 0 and (i_episode + 1) % (5 * print_progess_frequency) == 0:
                            print('epsilon:', self.epsilon)
                            print('predict_rewards:', self.training_context['train_data']['predict_rewards'][:5])
                            print('target_rewards:', self.training_context['train_data']['target_rewards'][:5])
                            print('reward_batch:', self.training_context['train_data']['reward_batch'][:5])
                            loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history,
                                              legend=['dqn'], calculate_base='epoch', imshow=imshow)

                    break

            # 定期更新target_net權值
            if start_train and i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)
                self.save_model(save_path=self.training_context['save_path'])

        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()
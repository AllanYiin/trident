import copy
import time
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
_session = get_session()
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
        super()._initial_graph(input_shape=tensor_to_shape(self.get_observation(),need_exclude_batch_axis=True,is_singleton=True),output=deepcopy(self.network))

    def get_observation(self):
        return self.data_preprocess(self.env.render('rgb_array'))

    def select_action(self, state,model_only=False, **kwargs):
        pass

    def get_rewards(self, action):
        observation_, reward, done, info = self.env.step(action.item())
        return reward

    def experience_replay(self):
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
        objecttype=None
        if isinstance(self.model.input_spec, TensorSpec):
            objecttype=self.model.input_spec.object_type
        #super()._initial_graph(inputs=to_tensor(self.get_observation()).repeat_elements(2, 0), output=deepcopy(self.network))
        self.setting_network()
        if objecttype is not None:
            self.inputs.value_list[0].object_type=objecttype
            self.model.input_spec.object_type = objecttype

        self.env.reset()
    def data_preprocess(self, img_data):
        if self._model is not None:
            self._model.input_spec.object_type=ObjectType.rgb
        if not hasattr(self,'_preprocess_flow') or self._preprocess_flow is None:
            self._preprocess_flow=[]
        if img_data.ndim==4:
            return to_tensor(to_numpy([self.data_preprocess(im) for im in img_data]))
        if len(self._preprocess_flow) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self._preprocess_flow:
                if self._model is not None and self.signature is not None and len(self.signature) > 1 and self._model.input_spec is not None:
                    img_data = fc(img_data,spec=self._model.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)
            if self._model.input_spec is None :
                self._model.input_spec= TensorSpec(shape=tensor_to_shape(to_tensor(img_data),need_exclude_batch_axis=True,is_singleton=True), object_type=ObjectType.rgb, name='input')

                self.input_shape=self._model.input_spec.shape[1:]

            return img_data
        else:
            return img_data

    def do_on_batch_end(self):
        self.training_context['time_batch_progress']+=( time.time() -self.training_context['time_batch_start'] )
        self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['steps']+=1
        if (self.training_context['steps']+1) % _session.epoch_equivalent == 0:
            if self.warmup > 0 and self.warmup == (self.training_context['steps']+1) // _session.epoch_equivalent:
                self.adjust_learning_rate(self.training_context['base_lr'])
                self.warmup = 0


class DqnPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 10000
                 , gamma=0.99, max_epsilon=0.9, min_epsilon=0.01, decay=100
                 , target_update=10, batch_size=10, name='dqn') -> None:
        super(DqnPolicy, self).__init__(network=network, env=env, memory_length=memory_length, name=name)

        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.steps_done = 0

    def setting_network(self):
        super()._initial_graph(input_shape=tensor_to_shape(self.get_observation(), need_exclude_batch_axis=True), output=deepcopy(self.network))
        self.policy_net = self.model
        self.policy_net.to(get_device())
        self.policy_net.train()

        self.target_net = copy.deepcopy(self.network)
        self.target_net.to(get_device())
        self.target_net.eval()
        self.summary()

    def get_observation(self):
        # 在這邊維護取得STATE的方法
        return np.expand_dims(np.array(list(self.env.state)), 0).astype(np.float32)

    def select_action(self, state, model_only=False, **kwargs):
        # 在這邊維護智能體如何選擇行動的邏輯
        # max_epsilon = 0.9  #初期還沒有案例可以供建模，因此大部分根據隨機案例
        # min_epsilon = 0.01 # 即使模型準確率越來越高，還是必須保留部分比例基於隨機案例
        # decay =200   # 衰減速度

        sample = random.random()
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1.0 * self.steps_done / self.decay)
        self.steps_done += 1
        if model_only == True or sample > self.epsilon:
            # 原始DQN是根據目標網路選取行動，而double DQN則是在策略網路中選取行動，更新模型時則參考目標網路
            with torch.no_grad():
                selected_action = expand_dims(argmax(self.policy_net(to_tensor(state)), 1), 0)
                return selected_action.item()
        else:
            selected_action = np.random.randint(low=0, high=100) % self.env.action_space.n
            return selected_action

    def get_rewards(self, action):
        # 在這邊維護取得獎賞的方法
        # 切換到下一時間點
        observation_, reward, done, info = self.env.step(action)

        # x, x_dot, theta, theta_dot = observation_
        # r1 = 0 if abs(x) < 0.5 * self.env.x_threshold else clip(abs(x) / self.env.x_threshold, 0, 1) - 0.5
        # r2 = 0 if abs(theta) < 0.5 * self.env.theta_threshold_radians else clip(abs(theta) / self.env.theta_threshold_radians, 0, 1) - 0.5
        #
        # reward = reward - 2 * r1 - 2 * r2 if not done else -3
        return reward, done

    def experience_replay(self):
        # 經驗回放
        # 在數據到位時，如何產生獎賞預測值，以及根據次一狀態產生未來獎賞的估計

        batch = self.memory.sample(self.batch_size)

        state_batch = to_tensor(batch.state, requires_grad=True).squeeze(1)
        action_batch = to_tensor(batch.action).long().detach()
        reward_batch = to_tensor(batch.reward).squeeze(1).detach()

        # 基於目前狀態所產生的獎賞期望值預測 Q(s_t)。
        # 策略網路評估行動
        predict_rewards = self.policy_net(state_batch).gather(1, action_batch)

        target_rewards = zeros(len(batch.next_state))
        # 至於未來狀態的部分，未來狀態主要是計算未來價值使用，但是只要是done，等於不再有未來狀態，則要視成功(1)是失敗(-3)來決定獎賞
        for i in range(len(batch.next_state)):
            s = batch.next_state[i]
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
        train_data['predict_rewards'] = predict_rewards
        train_data['target_rewards'] = expand_dims(target_rewards, 1)
        train_data['reward_batch'] = reward_batch

        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'predict_rewards'
        data_feed['target'] = 'target_rewards'

        self.training_context['data_feed'] = data_feed
        self.training_context['train_data'] = train_data

    def learn(self, num_episodes=300, batch_size=None, print_progess_frequency=10, min_replay_samples=50, repeat_train=16, imshow=True):
        # 智能體學習的主方法
        if batch_size is not None:
            self.batch_size = batch_size

        # 學習一開始將steps_done清零，逐步降低隨機決策比率
        self.steps_done = 0
        train_cnt = 0
        success_cnt = 0
        keep_success_cnt = 0
        start_train_episode = 0
        start_train = False
        # 收集初始資料
        while start_train == False:
            # 重置環境
            self.env.reset()
            # 獎賞清零
            total_rewards = 0
            state = self.get_observation()

            for t in count():
                # 基於目前狀態產生行動
                action = self.select_action(state, model_only=False)
                # 基於行動產生獎賞以及判斷是否結束(此時已經更新至下一個時間點)
                reward, done = self.get_rewards(action)

                # 累積獎賞
                total_rewards += reward

                # 任務完成強制終止(以300為基礎)
                conplete = (not done and t + 1 >= 300)

                if imshow:
                    # 更新視覺化螢幕
                    self.env.render()
                # 取得下一時間點觀察值
                next_state = None if done and not conplete else self.get_observation()

                # 將四元組儲存於記憶中
                # 如果要減少「好案例」的儲存比例請移除註解
                self.memory.push(state, action, next_state, reward)
                if len(self.memory) % 100 == 0:
                    print("Replay Samples:{0}".format(len(self.memory)))
                if len(self.memory) == min_replay_samples:
                    print('Start Train!!', flush=True)
                    # 需要記憶中的案例數大於批次數才開始訓練
                    start_train = (len(self.memory) >= min_replay_samples)
                    break

                # 切換至下一狀態
                state = copy.deepcopy(next_state)

                if done or conplete:
                    break

        # 開始訓練模式
        self.training_context['steps'] = 0
        self.steps_done = 0
        for i_episode in range(num_episodes):
            for i in range(repeat_train):
                # 經驗回放獲得訓練用批次數據
                self.output_fn = self.experience_replay

                # 訓練模型
                self.train_model(None, None,
                                 current_epoch=i_episode,
                                 current_batch=i,
                                 total_epoch=num_episodes,
                                 total_batch=repeat_train,
                                 is_collect_data=True if t >= 0 else False,
                                 is_print_batch_progress=False,
                                 is_print_epoch_progress=False,
                                 log_gradients=False, log_weights=False,
                                 accumulate_grads=False)

            # 定期更新target_net權值
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)
                self.save_model(save_path=self.training_context['save_path'])

            # 重置環境
            self.env.reset()
            # 獎賞清零
            total_rewards = 0
            state = self.get_observation()
            tmp_memory = []

            for t in count():
                # 透過優化器進行一步優化

                # 基於目前狀態產生行動
                action = self.select_action(state, model_only=True)
                # 基於行動產生獎賞以及判斷是否結束(此時已經更新至下一個時間點)
                reward, done = self.get_rewards(action)
                # 累積獎賞
                total_rewards += reward

                # 任務完成強制終止(以300為基礎)
                conplete = (not done and t + 1 >= 300)

                if imshow:
                    # 更新視覺化螢幕
                    self.env.render()
                # 取得下一時間點觀察值
                next_state = None if done else self.get_observation()

                # 將四元組儲存於記憶中
                tmp_memory.append((state, action, next_state, reward))

                # 切換至下一狀態
                state = next_state

                if done or conplete:
                    if t >= 200:
                        success_cnt += 1
                    else:
                        success_cnt = 0

                    # 判斷是否連續可達300分，如果是則停止學習

                    if t + 1 >= 300:
                        keep_success_cnt += 1
                    else:
                        keep_success_cnt = 0
                    if keep_success_cnt >= 2:
                        self.training_context['stop_update'] = 1
                    else:
                        self.training_context['stop_update'] = 0

                    # 紀錄累積獎賞
                    self.epoch_metric_history.collect('total_rewards', i_episode, float(total_rewards))
                    self.epoch_metric_history.collect('original_rewards', i_episode, float(t))
                    # 紀錄完成比率(以200為基礎)
                    self.epoch_metric_history.collect('task_complete', i_episode, 1.0 if t + 1 >= 200 else 0.0)
                    # 定期列印學習進度
                    if i_episode > 0 and i_episode % print_progess_frequency == 0:
                        self.print_epoch_progress(print_progess_frequency)
                    # 定期繪製損失函數以及評估函數對時間的趨勢圖
                    if i_episode > 0 and i_episode % (5 * print_progess_frequency) == 0:
                        print('negative_reward_ratio:', less(self.training_context['train_data']['reward_batch'], 0).mean().item())
                        print('predict_rewards:', self.training_context['train_data']['predict_rewards'].copy()[:5, 0])
                        print('target_rewards:', self.training_context['train_data']['target_rewards'].copy()[:5, 0])
                        print('reward_batch:', self.training_context['train_data']['reward_batch'].copy()[:5])
                        loss_metric_curve(self.epoch_loss_history, self.epoch_metric_history, legend=['dqn'], calculate_base='epoch', imshow=imshow)

                    if success_cnt == 50:
                        self.save_model(save_path=self.training_context['save_path'])
                        print('50 episodes success, training finish! ')
                        return True

                    break
            # print([item[3] for item in tmp_memory])
            sample_idx = []
            indexs = list(range(len(tmp_memory)))
            if len(tmp_memory) > 10:
                # 只保留失敗前的3筆以及隨機抽樣sqrt(len(tmp_memory))+5筆
                sample_idx.extend(indexs[-1 * min(3, len(tmp_memory)):])
                sample_idx.extend(random_choice(indexs[:-3], int(sqrt(len(tmp_memory)))))

            sample_idx = list(set(sample_idx))
            for k in range(len(tmp_memory)):
                state, action, next_state, reward = tmp_memory[k]
                if k in sample_idx or (k + 3 < len(tmp_memory) and tmp_memory[k + 1][3] < 1) or reward < 1:
                    self.memory.push(state, action, next_state, reward)

        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()




class PolicyGradientPolicy(PolicyBase):
    """The base class for any RL policy.
    """

    def __init__(self, network: Layer, env: gym.Env, memory_length: int = 100000, gamma=0.9, max_epsilon=0.9, min_epsilon=0.01, decay=200, target_update=10, batch_size=10,
                 name='pg') -> None:
        super(PolicyGradientPolicy, self).__init__(network=network, env=env, memory_length=memory_length, name=name)


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
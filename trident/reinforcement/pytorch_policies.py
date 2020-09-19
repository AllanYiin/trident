import copy
import math
import random
from itertools import count
import torch
import numpy as np
from matplotlib.pylab import plt
from torch import nn
from typing import Any, List, Union, Mapping, Optional, Callable
from trident.data.image_common import image_backend_adaption
from trident.backend.common import*
from trident.backend.pytorch_ops import *
from trident.backend.pytorch_backend import *
from trident.optims.pytorch_optimizers import Optimizer,get_optimizer
from trident.optims.pytorch_trainer import Model
from trident.optims.pytorch_losses import *
from trident.reinforcement.utils import ReplayBuffer,Transition
import_or_install('gym')
import gym

__all__ = ['PolicyBase', 'DqnPolicy']


class PolicyBase(Model):
    """The base class for any RL policy.
    """
    def __init__( self,network:Layer,env:gym.Env,memory_length:int=1000,name=None) -> None:

        self.env=env
        self.env.reset()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.network=network
        #self.agent_id = self.uuid
        self.memory=ReplayBuffer(memory_length)
        self.transform_funcs=[]
        super().__init__(inputs=self.get_observation(),output=copy.deepcopy(network))
        self.name = name



    def get_observation(self):
        return np.expand_dims(self.process_flow(self.env.render('rgb_array')),0)


    def select_action(self, state, **kwargs):
        pass


    def process_flow(self,data):
        if len(self.transform_funcs) == 0:
            return image_backend_adaption(data)
        if isinstance(data, np.ndarray):
            for fc in self.transform_funcs:
                if not fc.__qualname__.startswith(
                        'random_') or 'crop' in fc.__qualname__ or 'rescale' in fc.__qualname__ or (
                        fc.__qualname__.startswith('random_') and random.randint(0, 10) % 2 == 0):
                    data = fc(data)
            data = image_backend_adaption(data)
            return data

    def learn(self,num_episodes = 3000):
        pass




class DqnPolicy(PolicyBase):
    """The base class for any RL policy.
    """
    def __init__( self,network:Layer,env:gym.Env,memory_length:int=1000,gamma=0.999,episode_start=0.9,episode_end=0.05,episode_decay=200,target_update=10,batch_size=10,name='dqn') -> None:
        super(DqnPolicy, self).__init__(network=network,env=env,memory_length=memory_length,name=name)
        self.policy_net=self._model
        self.policy_net.train()
        self.target_net=copy.deepcopy(self._model)
        self.target_net.eval()
        summary(self.policy_net,tuple(self.inputs.value_list[0]))
        summary(self.target_net, tuple(self.inputs.value_list[0]))


        self.gamma=gamma
        self.episode_start=episode_start
        self.episode_end=episode_end
        self.episode_decay=episode_decay
        self.target_update=target_update
        self.batch_size=batch_size
        self.steps_done = 0


    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.episode_end + (self.episode_start - self.episode_end) * math.exp(-1. * self.steps_done / self.episode_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return argmax(self.policy_net(to_tensor(state))).view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space.n)]], device=get_device(), dtype=torch.long)

    def do_on_data_received(self, train_data, test_data):
        batch=train_data['batch']

        next_state_batch = to_tensor(batch.next_state).squeeze(1)
        state_batch =to_tensor( batch.state).squeeze(1)
        action_batch = to_tensor(batch.action).squeeze(1).long()
        reward_batch =to_tensor(batch.reward).squeeze(1)

        # Compute Q(s_t, a) - the model computes Q(s_t)
        # predict expected return of current state using main network
        predict_rewards= self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        q_next= max(self.target_net(next_state_batch),axis=1).detach()

        # Compute the expected Q values
        target_rewards =reward_batch+ (q_next * self.gamma)
        target_rewards=target_rewards.unsqueeze(1)
        train_data['state'] = state_batch
        train_data['predict_rewards']=predict_rewards
        train_data['target_rewards'] = target_rewards
        data_feed = OrderedDict()
        data_feed['input'] = 'state'
        data_feed['output'] = 'predict_rewards'
        data_feed['target'] = 'target_rewards'
        self.training_context['data_feed']=data_feed
        self.training_context['train_data'] = train_data
        return train_data, test_data




    def learn(self, num_episodes=3000, imshow=True):
        self._metrics = OrderedDict()
        self.epoch_metric_history= OrderedDict()
        self.epoch_metric_history['episode_durations']=[]
        self.epoch_metric_history['total_rewards'] = []
        self.epoch_loss_history['total_losses']=[]

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            self.steps_done = 0
            total_rewards=0
            last_screen=self.get_observation()
            current_screen=self.get_observation()
            state = current_screen - last_screen
            start_train=(len(self.memory)>self.batch_size)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                total_rewards+=reward
                reward = torch.tensor([reward], device=get_device())

                # Observe new state
                last_screen = current_screen
                current_screen =self.get_observation()
                if imshow:
                    self.env.render()
                    # plt.figure()
                    # plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                    # plt.title('screen')
                    # plt.show()
                next_state = current_screen - last_screen

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if start_train:
                    trainData=OrderedDict()
                    trainData['batch']=self.memory.sample(self.batch_size)
                    self.train_model(trainData, None,
                                                      current_epoch= i_episode,
                                                      current_batch=t,
                                                      total_epoch=num_episodes,
                                                      total_batch=t if done else t+1,
                                                      is_collect_data=True,
                                                      is_print_batch_progress=False,
                                                      is_print_epoch_progress=False,
                                                      log_gradients=False, log_weights=False,
                                                      accumulate_grads=False)

                if done:
                    self.epoch_metric_history['episode_durations'].append(t + 1)
                    self.epoch_metric_history['total_rewards'].append(total_rewards)
                    if start_train:
                        self.epoch_loss_history['total_losses'].append(np.array(self.training_context['losses']['total_losses'])[-t:].mean())
                        self.print_epoch_progress(1)

                    break
            # Update the target network
            if start_train and i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict(),strict=False)
                self.save_model(save_path=self.training_context['save_path'])
        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()
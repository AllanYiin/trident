import builtins
import base64
import datetime
import io
import os
import numpy as np
import cv2
from typing import Optional
import subprocess
from unittest.mock import patch
from collections import deque
from typing import Optional
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import animation

import gym
from gym import Wrapper
from gym import spaces
from gym.spaces import Box
from gym.wrappers import Monitor as _monitor
from gym.wrappers import LazyFrames
from matplotlib import animation
from pyvirtualdisplay import Display
from trident.reinforcement import ObservationType


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        #lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        #self.lives = self.env.unwrapped.ale.lives()
        self.lives = self.env.unwrapped._life
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame[34:194], (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class TimeAwareObservation(gym.ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory.
    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet.
    """
    def __init__(self, env, max_time=None):
        super(TimeAwareObservation, self).__init__(env)
        self.max_time=max_time
        if  isinstance(env.observation_space, Box) and env.observation_space.dtype == np.float32:
            low = np.append(self.observation_space.low, 0.0)
            high = np.append(self.observation_space.high, np.inf)
            self.observation_space = Box(low, high, dtype=np.float32)
            self.obsetvation_type=ObservationType.Box
        elif  isinstance(env.observation_space, Box) and env.observation_space.dtype == np.float32:
            self.obsetvation_type = ObservationType.Image

    def observation(self, observation):
        if self.obsetvation_type==ObservationType.Box:
            return np.append(observation, self.t if self.max_time is None or self.max_time==0 else self.t/float(self.max_time ))
        elif self.obsetvation_type==ObservationType.Image and self.max_time>0:
            ratio=self.t/float(self.max_time)
            pixels=builtins.round(self.observation_space.shape[1]*ratio)
            if len(self.observation_space.shape)==2:
                observation[-3:, :]=255
                observation[-3:, :pixels] = 0
            elif len(self.observation_space.shape) == 2:
                observation[-3:, :,:] =0
                observation[-3:, :, 0] = 255
                observation[-3:, :pixels,0] = 0
            return observation


    def step(self, action):
        self.t += 1
        return super(TimeAwareObservation, self).step(action)

    def reset(self, **kwargs):
        self.t = 0
        return super(TimeAwareObservation, self).reset(**kwargs)

class _VirtualDisplaySingleton(object):
    def __new__(cls,*args,**kwargs):
        if not hasattr(cls,"_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,size=(1024, 768)):
        self.size = size

        if not hasattr(self,"_display"):
            self._display = Display(visible=0,size=self.size)

            original = subprocess.Popen
            def Popen(cmd,pass_fds,stdout,stderr,shell):
                return original(cmd,pass_fds=pass_fds,
                                stdout=stdout,stderr=stderr,
                                shell=shell,preexec_fn=os.setpgrp)

            with patch("subprocess.Popen",Popen):
                self._display.start()

    def _restart_display(self):
        self._display.stop()
        self._display.start()


class VirtualDisplay(Wrapper):
    """
    Wrapper for running Xvfb
    """
    def __init__(self,env,size=(1024, 768)):
        """
        Wrapping environment and start Xvfb
        """
        super().__init__(env)
        self.size = size
        self._display = _VirtualDisplaySingleton(size)

    def render(self,mode=None,**kwargs):
        """
        Render environment
        """
        return self.env.render(mode='rgb_array',**kwargs)


class Monitor(_monitor):
    """
    Monitor wrapper to store images as videos.

    This class is a shin wrapper for `gym.wrappers.Monitor`. This class also
    have a method `display`, which shows recorded movies on Notebook.

    See Also
    --------
    gym.wrappers.Monitor : https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """
    def __init__(self,env,directory: Optional[str]=None,size=(1024, 768),
                 *args,**kwargs):
        """
        Initialize Monitor class

        Parameters
        ----------
        directory : str, optional
            Directory to store output movies. When the value is `None`,
            which is default, "%Y%m%d-%H%M%S" is used for directory.
        """
        if directory is None:
            directory = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._display = _VirtualDisplaySingleton(size)
        super().__init__(env,directory,*args,**kwargs)

    def _close_running_video(self):
        if self.video_recorder:
            self._close_video_recorder()
            self.video_recorder = None
        self._flush(force=True)

    def step(self,action):
        """
        Step Environment
        """
        try:
            return super().step(action)
        except KeyboardInterrupt as k:
            self._close_running_video()
            raise

    def reset(self,**kwargs):
        """
        Reset Environment
        """
        try:
            if self.stats_recorder and not self.stats_recorder.done:
                # StatsRecorder requires `done=True` before `reset()`
                self.stats_recorder.done = True
                self.stats_recorder.save_complete()

            return super().reset(**kwargs)
        except KeyboardInterrupt:
            self._close_running_video()
            raise

    def display(self,reset: bool=False):
        """
        Display saved all movies

        If video is running, stop and flush the current video then display all.

        Parameters
        ----------
        reset : bool, optional
            When `True`, clear current video list. This does not delete movie files.
            The default value is `False`, which keeps video list.
        """

        # Close current video.
        self._close_running_video()

        for f in self.videos:
            if not os.path.exists(f[0]):
                continue

            video = io.open(f[0], "r+b").read()
            encoded = base64.b64encode(video)

            display.display(os.path.basename(f[0]))
            display.display(display.HTML(data="""
            <video alt="test" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>
            """.format(encoded.decode('ascii'))))

        if reset:
            self.videos = []

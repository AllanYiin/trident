import builtins
import base64
import datetime
import io
import os
import random
import copy
import numpy as np
import shutil
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

from trident.backend.common import make_dir_if_need, get_plateform, get_time_suffix, sanitize_path, if_none, split_path, \
    OrderedDict
from trident.misc.ipython_utils import *
from trident.reinforcement.utils import ObservationType, ActionStrategy
from trident import context
from trident.data.vision_transforms import Resize
from trident.backend.pillow_backend import array2image, image2array

ctx = context._context()
__all__ = ['NoopResetEnv', 'EpisodicLifeEnv', 'MaxAndSkipEnv', 'RunningAvgAndSkipEnv', 'WarpFrame', 'FrameStack',
           'TimeAwareObservation', 'VideoRecording']


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, preferred_actions=None):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self.preferred_actions = preferred_actions
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0

        obs = None
        remain_noops = noops
        while remain_noops > 0:
            if random.random() < 0.8:
                if isinstance(self.preferred_actions, list) and len(self.preferred_actions) > 0:
                    obs, _, done, _ = self.env.step(random.choice(self.preferred_actions))
                else:
                    obs, _, done, _ = self.env.step(self.noop_action)
                remain_noops -= 1
                if done:
                    obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = self.env.unwrapped._life
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        # lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # ,so it's important to keep lives > 0, so that we only reset once
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

        # self.lives = self.env.unwrapped.ale.lives()
        self.lives = self.env.unwrapped._life
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = None
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


class RunningAvgAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((skip,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)

            self._obs_buffer[i] = obs

            total_reward += reward
            if done or self.env.unwrapped._is_dying or self.env.unwrapped._is_dead or self.env.unwrapped._life < 2:
                done = True
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        weight = np.reshape(np.array([1, 2, 3, 4]), (4, 1, 1, 1))
        max_frame = (weight * self._obs_buffer).sum(axis=0) / 10.0

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
        frame = cv2.resize(frame[34:194], (self._width, self._height), interpolation=cv2.INTER_AREA)
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                            dtype=env.observation_space.dtype)

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
    """Augment the observation with current time step in the trajectory.
    Currently, it only works with one-dimensional observation space.
    It doesn't support pixel observation space yet.
    """

    def __init__(self, env, max_time=None):
        super(TimeAwareObservation, self).__init__(env)
        self.stay_fool = False
        if max_time is None:
            self.max_time = env.unwrapped._time_last
        else:
            self.max_time = max_time

        if isinstance(env.observation_space, Box) and ((
                                                               env.observation_space.low.max() == 0 and env.observation_space.high.min() == 255) and env.observation_space.dtype in [
                                                           np.uint8, np.float32]):
            self.obsetvation_type = ObservationType.Image
        elif isinstance(env.observation_space, Box) and env.observation_space.dtype == np.float32:
            low = np.append(self.observation_space.low, 0.0)
            high = np.append(self.observation_space.high, np.inf)
            self.observation_space = Box(low, high, dtype=np.float32)
            self.obsetvation_type = ObservationType.Box

    def observation(self, observation):

        if self.obsetvation_type == ObservationType.Box:
            return np.append(observation,
                             self.t if self.max_time is None or self.max_time == 0 else self.t / float(self.max_time))
        elif self.obsetvation_type == ObservationType.Image and self.max_time > 0:
            ratio = 0
            if hasattr(super(TimeAwareObservation, self).unwrapped, '_time'):
                ratio = (self.max_time - super(TimeAwareObservation, self).unwrapped._time) / float(self.max_time)
            original_dtype = observation.dtype
            pixels = builtins.round(self.observation_space.shape[1] * ratio)

            H, W, C = observation.shape
            _observation = observation.copy().astype(np.float32)
            obj_base = np.zeros((W, W, C)).astype(np.float32)
            obj_base[:H, :, :] = _observation
            if self.stay_fool or ratio > 0.5:
                noise_mask = np.ones((1, W, 1))
                noise_mask[:, :W // 4, :] = np.expand_dims(np.expand_dims(np.linspace(0.2, 1, W // 4), 0), -1).astype(
                    np.float32)
                noise_mask = np.concatenate([noise_mask, noise_mask, noise_mask], axis=-1)
                obj_base = np.clip(
                    obj_base * noise_mask + (1 - noise_mask) * (80 * np.random.standard_normal((W, W, C))), 0, 255)

            if len(self.observation_space.shape) == 2:
                obj_base[-5:, :pixels] = 0
                obj_base[-5:, pixels:] = 255
            elif len(self.observation_space.shape) == 3:
                obj_base[-5:, :pixels, :] = 0
                if self.stay_fool:
                    obj_base[-5:, pixels:, :] = 255
                    obj_base[-5:, pixels:, 1:] = 0
                else:
                    obj_base[-5:, pixels:, :] = 255
            return np.clip(obj_base, 0, 255).astype(original_dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'stay_fool' in info:
            self.stay_fool = info['stay_fool']
        time_awareness_obj = self.observation(observation)
        return time_awareness_obj, reward, done, info

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    # def render(self, mode='human', **kwargs):
    #     if mode=='human':
    #         return super(TimeAwareObservation, self).render(mode, **kwargs)
    #     elif mode == 'observation':
    #         return self.observation(super(TimeAwareObservation, self).render('rgb_array').copy())
    #     elif mode=='rgb_array':
    #         return super(TimeAwareObservation, self).render('rgb_array')


class _VirtualDisplaySingleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, size=(1024, 768)):
        self.size = size

        if not hasattr(self, "_display"):
            self._display = Display(visible=0, size=self.size)
            original = subprocess.Popen

            def Popen(cmd, pass_fds, stdout, stderr, shell):
                return original(cmd, pass_fds=pass_fds,
                                stdout=stdout, stderr=stderr,
                                shell=shell, preexec_fn=os.setpgrp)

            with patch("subprocess.Popen", Popen):
                self._display.start()

    def _restart_display(self):
        self._display.stop()
        self._display.start()


class VirtualDisplay(Wrapper):
    """
    Wrapper for running Xvfb
    """

    def __init__(self, env, size=(1024, 768)):
        """
        Wrapping environment and start Xvfb
        """
        super().__init__(env)
        self.size = size
        self._display = _VirtualDisplaySingleton(size)

    def render(self, mode=None, **kwargs):
        """
        Render environment
        """
        return self.env.render(mode='rgb_array', **kwargs)


class VideoRecording(gym.Wrapper):
    """
    Monitor wrapper to store images as videos.

    This class is a shin wrapper for `gym.wrappers.Monitor`. This class also
    have a method `display`, which shows recorded movies on Notebook.

    See Also
    --------
    gym.wrappers.Monitor : https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def __init__(self, env, directory: Optional[str] = None, enabled=False, fps=None, min_frames=None,
                 done_then_finish=True, name_prefix=None, **kwargs):
        """
        Initialize Monitor class

        Parameters
        ----------
        directory : str, optional
            Directory to store output movies. When the value is `None`,
            which is default, "%Y%m%d-%H%M%S" is used for directory.
        """
        gym.Wrapper.__init__(self, env)
        self._recording_enabled = enabled
        self._is_recording = False
        self.num_frames = 0
        self.directory = directory
        self.done_then_finish = done_then_finish
        self.video_tags = OrderedDict()
        if directory is None:
            self.directory = make_dir_if_need('videos')
        else:
            self.directory = make_dir_if_need(sanitize_path(directory))
        self.fps = fps
        if 'video.frames_per_second' not in self.env.metadata:
            self.env.metadata['video.frames_per_second'] = self.fps if self.fps is not None else 30
        self.min_frames = min_frames
        self.videos = []
        self._display = None
        shp = env.observation_space.shape
        if get_plateform() != 'windows':
            self._display = _VirtualDisplaySingleton((shp[1], shp[0]))
        if name_prefix is None:
            name_prefix = 'video'

        self.name_prefix = name_prefix
        self.current_recording_path = None
        self.vw = None
        # self.resize = Resize((84, 84), keep_aspect=True)
        # self.vw2 = None
        self.frame_steps = 0
        self.prev_screen = None
        self.current_screen = None
        self.prev_observation = None
        self.current_observation = None
        self.prev_reward = 0
        self.prev_done = False
        self.info = None
        self.frame_stats = []

    def create_video_writer(self):
        shp = (240, 240, 3)  # super().render(mode='rgb_array').shape
        video_name = os.path.join(self.directory, self.name_prefix + '_' + get_time_suffix() + '.avi')
        # video_name2 = os.path.join(self.directory, self.name_prefix + '_obj_' + get_time_suffix() + '.avi')
        self.current_recording_path = video_name
        # self.current_recording_path2 = video_name2

        self.vw = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  if_none(self.fps, self.env.metadata['video.frames_per_second']), (shp[1], shp[0]))
        self.video_tags = OrderedDict()

        # self.vw2 = cv2.VideoWriter(video_name2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.env.metadata['video.frames_per_second'], (84*4, 84*4))

        self._is_recording = True
        self.frame_steps = 0

    def close_video_recorder(self):

        if self._is_recording and (self.vw is not None and self.frame_steps > 0):
            self.vw.release()
            self.vw = None
            # self.vw2.release()
            self._is_recording = False

            if 'keep' in self.video_tags and self.video_tags['keep']:
                if 'name_suffix' in self.video_tags:
                    folder, filename, ext = split_path(self.current_recording_path)
                    os.rename(self.current_recording_path,
                              os.path.join(folder, filename + '_' + self.video_tags['name_suffix'] + ext))
                self.videos.append(self.current_recording_path)
            elif 'keep' in self.video_tags and not self.video_tags['keep']:
                os.remove(self.current_recording_path)
            elif (self.min_frames is not None and self.frame_steps < self.min_frames):
                os.remove(self.current_recording_path)
                #   os.remove(self.current_recording_path2 )
            else:
                self.videos.append(self.current_recording_path)

    def step(self, action):
        """
        Step Environment
        """

        try:

            observation, reward, done, info = self.env.step(action)

            self.done = done

            new_render = self.render('rgb_array').copy()

            self.prev_screen = copy.deepcopy(self.current_screen)
            self.current_screen = new_render
            self.prev_observation = copy.deepcopy(self.current_observation)
            self.current_observation = observation.copy()

            if self.current_observation.shape != self.prev_observation.shape:
                self.prev_observation = self.current_observation.copy()

            if self.vw is None:
                self.create_video_writer()

            # if self.done_then_finish and self.done:
            #     self.close_video_recorder()

            else:
                if self._recording_enabled and self._is_recording:
                    self.vw.write(cv2.cvtColor(self.current_observation, cv2.COLOR_RGB2BGR))
                    self.frame_steps += 1
                    self.env.unwrapped.frame_steps = self.frame_steps
                    # obj_frame= cv2.resize(cv2.cvtColor(observation.copy().copy(), cv2.COLOR_RGB2BGR),(84*4,84*4),cv2.INTER_LANCZOS4)
                    # self.vw2.write(obj_frame)

            self.prev_reward = reward
            self.prev_done = done
            self.info = info

            self.state = observation
            return self.state.copy(), reward, done, info
        except KeyboardInterrupt as k:
            self.close_video_recorder()
            raise

    def reset(self, **kwargs):
        """
        Reset Environment
        """
        try:

            if self.done_then_finish and self._is_recording and self.frame_steps > 0:
                self.close_video_recorder()
            if self._recording_enabled and (not self._is_recording or self.vw is None):
                self.create_video_writer()

            self.frame_stats = []
            self.prev_reward = 0
            self.prev_done = False
            self.info = None
            self.frame_steps = 0
            self.env.unwrapped.frame_steps = self.frame_steps
            observation = self.env.reset()

            self.prev_observation = observation
            self.current_observation = observation
            self.state = observation

            return self.state
        except KeyboardInterrupt:
            self.close_video_recorder()
            raise

    def display(self, reset: bool = False):
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
        if is_in_ipython():
            for f in self.videos:
                if not os.path.exists(f):
                    continue

                video = io.open(f[0], "r+b").read()
                encoded = base64.b64encode(video)

                display.display(os.path.basename(f))
                display.display(display.HTML(data="""
                <video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>
                """.format(encoded.decode('ascii'))))

        if reset:
            self.videos = []

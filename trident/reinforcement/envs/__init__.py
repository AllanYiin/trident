import threading
from multiprocessing import Pipe
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp
from trident import context

ctx=context._context()

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))

        elif cmd == 'render':
            remote.send(env.render())

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError

class SubprocVecEnv():
    def __init__(self):
        self.waiting = False

        self.closed = False
        no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(no_of_envs)])
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target=worker,
                           args=(wrk, rem, CloudpickleWrapper(fn)))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        if self.waiting:
            raise  AlreadySteppingError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), info

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class MultipleEnvironments:
    def __init__(self,*envs,num_envs,**kwargs):
        if num_envs is None:
            num_envs=len(envs)
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])

        self.envs = list(envs)
        self.num_states = self.envs[0].observation_space.shape[0]

        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
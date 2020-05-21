import time

import torch
from torch.autograd import Function
from torch.autograd import Variable

"""
https://raw.githubusercontent.com/zhuwenxi/pytorch-profiling-tool/master/profiling.py
"""

class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.record = {'forward': [], 'backward': []}
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False
        self.layer_num = 0

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        ret = ""

        iter = len(self.record['forward']) / self.layer_num

        for i in xrange(iter):
            ret += "\n================================= Iteration {} =================================\n".format(i + 1)

            ret += "\nFORWARD TIME:\n\n"
            for j in xrange(self.layer_num):
                record_item = self.record['forward'][i * self.layer_num + j]
                ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1, record_item[2] - record_item[1],
                                                                               record_item[0])

            ret += "\nBACKWARD TIME:\n\n"
            for j in (xrange(self.layer_num)):
                record_item = self.record['backward'][i * self.layer_num + self.layer_num - j - 1]
                try:
                    ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1,
                                                                                   record_item[2] - record_item[1],
                                                                                   record_item[0])
                except:
                    # Oops, this layer doesn't execute backward post-hooks
                    pass

        return ret

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)

        self.profiling_on = True

        return self

    def stop(self):
        self.profiling_on = False

        return self

    def hook_modules(self, module):

        this_profiler = self

        sub_modules = module.__dict__['_modules']

        for name, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            if isinstance(sub_module, torch.nn.Container) or isinstance(sub_module, torch.nn.Sequential):
                #
                # nn.Container or nn.Sequential who have sub nn.Module. Recursively visit and hook their decendants.
                #
                self.hook_modules(sub_module)
            else:

                self.layer_num += 1

                #
                # nn.Module who doesn't have sub nn.Module, hook it.
                #

                # Wrapper function to "__call__", with time counter in it.
                def wrapper_call(self, *input, **kwargs):
                    start_time = time.time()
                    result = this_profiler.origin_call[self.__class__](self, *input, **kwargs)
                    stop_time = time.time()

                    that = self

                    def backward_pre_hook(*args):
                        if (this_profiler.profiling_on):
                            this_profiler.record['backward'].append((that, time.time()))

                    result.grad_fn.register_pre_hook(backward_pre_hook);

                    if (this_profiler.profiling_on):
                        global record
                        this_profiler.record['forward'].append((self, start_time, stop_time))

                    return result

                # Replace "__call__" with "wrapper_call".
                if sub_module.__class__ not in this_profiler.origin_call:
                    this_profiler.origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
                    sub_module.__class__.__call__ = wrapper_call

                def backward_post_hook(*args):
                    if (this_profiler.profiling_on):
                        this_profiler.record['backward'][-1] = (
                        this_profiler.record['backward'][-1][0], this_profiler.record['backward'][-1][1], time.time())

                sub_module.register_backward_hook(backward_post_hook)
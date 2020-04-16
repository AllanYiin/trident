import os
import sys
import time
from shutil import copyfile
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import  Optimizer
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.hooks as hooks
from collections import OrderedDict,defaultdict
from functools import partial
import numpy as np
from ..backend.common import get_session,addindent,get_time_prefix,get_class,format_time,get_terminal_size,snake2camel
from ..backend.pytorch_backend import *


_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len =max(int(TOTAL_BAR_LENGTH*float(current)/total),1)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('..')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    sys.stdout.write(' ( %d/%d )' % (current, total))
    sys.stdout.write('\n')
    sys.stdout.flush()
    # # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    # sys.stdout.write(' %d/%d ' % (current+1, total))
    # if current < total-1:
    #     sys.stdout.write('\r')
    # else:
    #     sys.stdout.write('\n')
    # sys.stdout.flush()


class TrainingItem(object):
    def __init__(self,model:nn.Module,optimizer,**kwargs):

        self.model= model
        if isinstance(optimizer,str):
            optimizer=get_class(optimizer,['torch.optim'])
        self.optimizer=optimizer(self.model.parameters(),**kwargs)

        self._losses = OrderedDict()
        self._metrics = OrderedDict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''
    def __str__(self):
        self.__repr__()
    def _get_name(self):
        return self.__class__.__name__
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, value in self.__dict__.items():
            if isinstance(value,OrderedDict):
                for subkey, subvalue in value.items():
                    mod_str = repr(subvalue)
                    mod_str = addindent(mod_str, 2)
                    child_lines.append('(' + key + '): ' + mod_str)
            else:
                mod_str = repr(value)
                mod_str = addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())

        losses = list(self._losses.keys())
        metrics = list(self._metrics.keys())
        keys = module_attrs + attrs  + losses+metrics

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def get_weight_reg(self, model):
        L1_reg = torch.tensor(0., requires_grad=True)
        L2_reg = torch.tensor(0, requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)
                L2_reg += param.norm(2)
        return L1_reg, L2_reg


    @classmethod
    def create(cls):
        plan = cls()
        return plan


    def with_loss(self, loss,  **kwargs):
        if hasattr(loss,'forward'):
            self._losses[loss.__name__] = loss(**kwargs)
        elif callable(loss):
            self._losses[loss.__name__] = partial(loss,**kwargs)
        return self
    def with_metrics(self, metrics,  **kwargs):
        if hasattr(metrics,'forward'):
            self._metrics[metrics.__name__] = metrics(**kwargs)
        elif callable(metrics):
            self._metrics[metrics.__name__] = partial(metrics, **kwargs)
        return self


class TrainingPlan(object):
    def __init__(self):
        self._training_items = OrderedDict()

        self._dataloaders = OrderedDict()

        self.num_epochs=1
        self.minibatch_size=1
        self.print_progress_frequency=10
        self.save_model_frequency=50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __getattr__(self, name):
        if name=='self':
            return self
        if '_training_items' in self.__dict__:
            _training_items = self.__dict__['_training_items']
            if name in _training_items:
                return _training_items[name]

        if '_dataloaders' in self.__dict__:
            _dataloaders = self.__dict__['_dataloaders']
            if name in _dataloaders:
                return _dataloaders[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))


    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''
    def __str__(self):
        self.__repr__()
    def _get_name(self):
        return self.__class__.__name__
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, value in self.__dict__.items():
            if isinstance(value,OrderedDict):
                for subkey, subvalue in value.items():
                    mod_str = repr(subvalue)
                    mod_str = addindent(mod_str, 2)
                    child_lines.append('(' + key + '): ' + mod_str)
            else:
                mod_str = repr(value)
                mod_str = addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        training_items = list(self._training_items.keys())
        keys = module_attrs + attrs + training_items

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    @classmethod
    def create(cls):
        plan = cls()
        return plan

    def add_training_item(self, training_item:TrainingItem):
        training_item.model.to(self.device)
        self._training_items[training_item.__repr__()]=training_item
        return self

    def with_data_loader(self, data_loader,  **kwargs):
        self._dataloaders[data_loader.__class__.__name__]=data_loader
        return self

    def repeat_epochs(self, num_epochs:int):
        self.num_epochs = num_epochs
        return self
    def within_minibatch_size(self, minibatch_size:int):
        self.minibatch_size = minibatch_size
        for i, (k, v) in enumerate(self._dataloaders.items()):
            v.minibatch_size=minibatch_size
            self._dataloaders[k]=v
        return self
    def print_progress_every(self, num_minibatch:int):
        self.print_progress_frequency= num_minibatch
        return self
    def save_model_every(self, num_minibatch:int,save_path:str ):
        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except Exception as e:
                sys.stderr.write(e.__str__())
        self.save_path=save_path
        self.save_model_frequency= num_minibatch
        return self


    def start_now(self, ):
        self.execution_id=str(uuid.uuid4())[:8].__str__().replace('-','')
        trainingitem=list(self._training_items.items())[0][1]
        data_loader=list(self._dataloaders.items())[0][1]
        loss_fn =    trainingitem._losses

        metrics_fn =trainingitem._metrics
        model_in_train=trainingitem.model
        optimizer=trainingitem.optimizer
        losses=[]
        metrics=[]
        print(self.__repr__)

        for epoch in range(self.num_epochs):
            try:
                for mbs,( input, target) in enumerate(data_loader):
                    input, target = torch.from_numpy(input),  torch.from_numpy(target)
                    input, target = Variable(input).to(self.device), Variable(target).to(self.device)
                    output =model_in_train(input)
                    current_loss =0
                    current_metrics={}
                    for k,v in loss_fn.items():
                        current_loss +=v.forward(output,target) if hasattr(v,'forward') else v(output,target)
                    for k, v in metrics_fn.items():
                        current_metrics[k]=to_numpy(v.forward(output,target) if hasattr(v,'forward') else v(output,target))

                    losses.append(to_numpy(current_loss))
                    metrics.append(current_metrics)

                    optimizer.zero_grad()
                    current_loss.backward()
                    optimizer.step()


                    if (mbs) % self.print_progress_frequency == 0:
                        base_metrics={}
                        for item in metrics:
                            for k,v in item.items():
                                if k not in base_metrics:
                                    base_metrics[k]=v
                                else:
                                    base_metrics[k]+=v

                        progress_bar(mbs, len(data_loader.batch_sampler), 'Loss: {0:.3f} | {1}'.format(np.array(losses).mean(), ','.join(['{0}: {1:.3%}'.format(snake2camel(k), v / len(metrics)) for k, v in base_metrics.items()])))
                        # print("Baseline:     Epoch: {}/{} Step: {} ".format(epoch + 1, self.num_epochs,mbs),
                        #       "Loss: {:.4f}...".format(np.array(losses).mean()),
                        #       ','.join(['{0}: {1:.3%}'.format(k, v / len(metrics)) for k, v in base_metrics.items()]))
                        # f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format(
                        #     'lanet_model', 0.01, epoch, mbs + 1, np.array(losses).mean(), np.array(metrics).mean())])

                        losses = []
                        metrics = []
                    if (mbs + 1) %self.save_model_frequency == 0:
                        # torch.save(lenet_model.state_dict(),'lenet_model_pytorch_mnist_1.pth' )
                        # torch.save(gcd_model.state_dict(), 'gcd_model_pytorch_mnist_1.pth')

                        save_full_path=os.path.join(self.self.save_path,'model_{0}_{1}_{2}.pth'.format(model_in_train.__class__.__name__,self.execution_id,get_time_prefix()))
                        torch.save(model_in_train,save_full_path)
                        #copyfile(src, dst)
                    if(mbs+1)%len(data_loader.batch_sampler)==0:
                        break
            except StopIteration:
                pass
            except ValueError as ve:
                print(ve)
            except Exception as e:
                print(e)


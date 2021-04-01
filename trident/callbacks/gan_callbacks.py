from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import os
import random
import sys
import warnings

import numpy as np
from trident.reinforcement.utils import ReplayBuffer

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import CallbackBase
from trident.data.vision_transforms import *
from trident.data.dataset import *
from trident.misc.visualization_utils import *

if get_backend() == 'pytorch':
    import torch
    import torch.nn as nn

    from trident.backend.pytorch_ops import to_numpy, to_tensor, shuffle, random_choice
    from trident.optims.pytorch_losses import CrossEntropyLoss, MSELoss, L1Loss, L2Loss, BCELoss
    from trident.optims.pytorch_constraints import min_max_norm
    from trident.optims.pytorch_trainer import *
    from trident.layers.pytorch_activations import *
    from trident.backend.pytorch_ops import *
    from trident.models.pytorch_efficientnet import EfficientNetB0
elif get_backend() == 'tensorflow':

    from trident.backend.tensorflow_ops import to_numpy, to_tensor
    from trident.optims.tensorflow_losses import CrossEntropyLoss, MSELoss, L1Loss, L2Loss, BCELoss
    from trident.optims.tensorflow_constraints import min_max_norm
    from trident.optims.tensorflow_trainer import *

__all__ = ['GanCallbacksBase', 'GanCallback', 'CycleGanCallback']


class GanCallbacksBase(CallbackBase):
    def __init__(self):
        super(GanCallbacksBase, self).__init__(is_shared=True)

    pass


def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt.mean()


class GanCallback(GanCallbacksBase):

    def __init__(self, generator=None, discriminator=None, gan_type='gan', label_smoothing=False, noisy_labels=False,
                 noised_real=True, noise_intensity=0.05, weight_clipping=False, tile_image_frequency=100,
                 experience_replay=False, use_total_variation=False, use_ttur=False,  # two timel-scale update rule
                 g_train_frequency=1, d_train_frequency=1, noised_lr=False, diversity_constraint=False, teacher_distill=False, **kwargs):
        _available_gan_type = ['gan', 'began', 'ebgan', 'wgan', 'wgan-gp', 'lsgan', 'lsgan1', 'rasgan']
        super(GanCallback, self).__init__()
        if isinstance(generator, ImageGenerationModel):
            self.generator = generator.model
        if isinstance(discriminator, ImageClassificationModel):
            self.discriminator = discriminator.model
        self.training_items = None
        self.data_provider = None
        self.data_feed = OrderedDict()
        self.train_data = OrderedDict()
        # self.z_noise = None
        # self.D_real = None
        # self.D_fake = None
        # self.D_metric = None
        # self.G_metric = None
        # self.img_real = None
        # self.img_fake = None
        self.gan_type = gan_type if gan_type in _available_gan_type else None
        self.label_smoothing = label_smoothing
        self.noisy_labels = noisy_labels
        self.noised_real = noised_real
        self.noise_intensity = noise_intensity
        self.tile_image_frequency = tile_image_frequency
        self.weight_clipping = weight_clipping
        self.experience_replay = experience_replay
        self.g_train_frequency = g_train_frequency
        self.d_train_frequency = d_train_frequency
        if self.experience_replay == True:
            make_dir_if_need('Replay')
        self.tile_images = []
        self.use_total_variation = use_total_variation
        self.generator_first = None
        self.cooldown_counter = 0
        self.beginning_repository = ReplayBuffer(capacity=250)
        self.latter_repository = ReplayBuffer(capacity=250)
        self.generator_worse_metric = None
        self.discriminator_worse_metric = None
        self.generator_best_metric = None
        self.discriminator_best_metric = None
        self.generator_best_epoch = None
        self.discriminator_best_metric = None
        self.noise_end_epoch = 20
        self.noised_lr = noised_lr
        self.diversity_constraint = diversity_constraint
        self.teacher_distill = teacher_distill
        if self.diversity_constraint == True or self.teacher_distill == True:
            self.effnetb0 = EfficientNetB0(pretrained=True, include_top=False)
            self.effnetb0.eval()
            self.effnetb0.model.trainable = False

    def on_training_start(self, training_context):
        self.training_items = training_context['training_items']
        self.data_provider = training_context['_dataloaders'].value_list[0]
        dses = self.data_provider.traindata.get_datasets()
        for k, training_item in self.training_items.items():
            if self.generator is not None and training_item.model.uuid == self.generator.uuid:
                training_item.training_context['gan_role'] = 'generator'
            elif self.discriminator is not None and training_item.model.uuid == self.discriminator.uuid:
                training_item.training_context['gan_role'] = 'discriminator'
            elif self.generator is None:
                raise ValueError('You need generator in gan model')
            elif self.discriminator is None:
                raise ValueError('You need discriminator in gan model')

            model = training_item.model
            if training_item.training_context['gan_role'] == 'generator':
                if not 'data_feed' in training_item.training_context:
                    training_item.training_context['data_feed'] = OrderedDict()
                for ds in dses:
                    if isinstance(ds, RandomNoiseDataset):
                        training_item.training_context['data_feed']['input'] = ds.symbol
                training_item.training_context['data_feed']['output'] = 'd_fake'
                training_item.training_context['data_feed']['target'] = 'label_fake'

            elif training_item.training_context['gan_role'] == 'discriminator':
                if not 'data_feed' in training_item.training_context:
                    training_item.training_context['data_feed'] = OrderedDict()
                training_item.training_context['data_feed']['input'] = 'd_input'
                training_item.training_context['data_feed']['output'] = 'd_output'
                training_item.training_context['data_feed']['target'] = 'd_label'

        if self.training_items.value_list[0].training_context['gan_role'] == 'generator':
            self.generator_first = True
            self.discriminator.training_context['retain_graph'] = True
            print('generator first')
        else:
            self.generator_first = False
            print('discriminator first')

    def on_data_received(self, training_context):
        try:
            traindata = training_context['train_data']
            traindata['img_real'] = traindata[self.data_provider.traindata.data.symbol]
            if self.generator_first == True:
                traindata['d_fake'] = self.discriminator(to_tensor(to_numpy(self.img_fake)))
            else:

                if 'img_fake' not in traindata or traindata['img_fake'] is None:
                    traindata['img_fake'] = random_normal_like(traindata['img_real'])

                traindata['img_fake']

                #
                # if self.experience_replay:
                #     self.img_fake = self.beginning_repository.push_and_pop(self.img_fake)
                #
                # if self.noisy_labels and training_context['current_epoch'] < self.noise_end_epoch:
                #     exchange_real = random_choice(self.img_real).clone()
                #     exchange_fake = random_choice(self.img_fake).clone()
                #     self.img_fake[random.choice(range(self.img_fake.size(0)))] = exchange_real
                #
                # if self.noised_real and training_context['current_epoch'] < self.noise_end_epoch and random.randint(0,100) % 10 < training_context['current_epoch']:
                #     self.img_real = (training_context['current_input'] + to_tensor(
                #         0.2 * (1 - float(curr_epochs) / self.noise_end_epoch) * np.random.standard_normal(
                #             list(self.img_real.size())))).clamp_(-1, 1)
                #
                # self.D_real = self.discriminator(self.img_real)
                # if not self.generator_first:
                #     self.D_fake = self.discriminator(to_tensor(to_numpy(self.img_fake)))
                # else:
                #     self.D_fake = self.discriminator(self.img_fake)
        except:
            PrintException()

    def on_loss_calculation_start(self, training_context):
        traindata = training_context['train_data']

        if training_context['gan_role'] == 'generator':
            data_feed = training_context['data_feed']
            traindata['img_fake'] = traindata[data_feed['output']]
            traindata['d_fake'] = self.discriminator(traindata['img_fake'])
            traindata['d_real'] = self.discriminator(traindata['img_real'])
        elif training_context['gan_role'] == 'discriminator':
            traindata['d_fake'] = self.discriminator(traindata['img_fake'])
            if not self.generator_first:
                traindata['d_real'] = self.discriminator(traindata['img_real'])
                traindata['d_fake'] = self.discriminator(traindata['img_fake'])

    def on_loss_calculation_end(self, training_context):
        is_collect_data = training_context['is_collect_data']

        true_label = to_tensor(np.ones((self.D_real.size()), dtype=np.float32))
        false_label = to_tensor(np.zeros((self.D_real.size()), dtype=np.float32))

        if self.label_smoothing:
            if training_context['current_epoch'] < 20:
                true_label = to_tensor(np.random.randint(80, 100, (self.D_real.size())).astype(np.float32) / 100.0)
            elif training_context['current_epoch'] < 50:
                true_label = to_tensor(np.random.randint(85, 100, (self.D_real.size())).astype(np.float32) / 100.0)
            elif training_context['current_epoch'] < 200:
                true_label = to_tensor(np.random.randint(90, 100, (self.D_real.size())).astype(np.float32) / 100.0)
            else:
                pass
        # true_label.requires_grad=False
        # false_label.requires_grad=False

        if training_context['gan_role'] == 'generator':

            if self.diversity_constraint == True or self.teacher_distill == True:
                try:
                    embedded_fake = self.effnetb0.model(self.img_fake)
                    embedded_fake = reshape(embedded_fake, [embedded_fake.shape[0], -1])
                    if self.diversity_constraint == True:
                        pl_loss = 0.1 * pullaway_loss(embedded_fake)
                        training_context['current_loss'] = training_context['current_loss'] + pl_loss
                        if is_collect_data:
                            if 'pullaway_loss' not in training_context['losses']:
                                training_context['losses']['pullaway_loss'] = []
                            training_context['losses']['pullaway_loss'].append(float(to_numpy(pl_loss)))
                    if self.teacher_distill == True:
                        try:

                            embedded_fake = embedded_fake.mean(dim=0)
                            embedded_real = self.effnetb0.model(self.img_real)
                            embedded_real = reshape(embedded_real, [embedded_real.shape[0], -1]).mean(dim=0).detach()
                            distill_loss = ((embedded_fake - embedded_real).abs()).mean()
                            training_context['current_loss'] = training_context['current_loss'] + distill_loss
                            if is_collect_data:
                                if 'distill_loss' not in training_context['losses']:
                                    training_context['losses']['distill_loss'] = []
                                training_context['losses']['distill_loss'].append(float(to_numpy(distill_loss)))
                        except:
                            PrintException()

                except:
                    PrintException()

            try:

                this_loss = 0
                if self.use_total_variation:
                    self.D_real = self.D_real.clamp(min=-1, max=1)
                    self.D_fake = self.D_fake.clamp(min=-1, max=1)

                if self.gan_type == 'gan':
                    adversarial_loss = torch.nn.BCELoss()
                    this_loss = adversarial_loss(self.D_fake, true_label)
                elif self.gan_type == 'dcgan':
                    adversarial_loss = torch.nn.BCELoss()
                    this_loss = adversarial_loss(self.D_fake, true_label)
                elif self.gan_type == 'wgan':
                    this_loss = -torch.mean(self.D_fake)

                elif self.gan_type == 'wgan-gp':
                    this_loss = -torch.mean(self.D_fake)

                elif self.gan_type == 'wgan-div':
                    this_loss = -torch.mean(self.D_fake)

                elif self.gan_type == 'lsgan':  # least squared
                    this_loss = torch.mean((self.D_fake - 1) ** 2)
                elif self.gan_type == 'lsgan1':  # loss sensitive
                    this_loss = torch.mean((self.D_fake - 1) ** 2)
                elif self.gan_type == 'rasgan':
                    D_fake_logit = sigmoid(self.D_fake - self.D_real.mean(0, True))

                    self.G_metric = ((1 - D_fake_logit) ** 2).mean()
                    if 'D_fake_logit' not in training_context['tmp_metrics']:
                        training_context['tmp_metrics']['D_fake_logit'] = []
                        training_context['metrics']['D_fake_logit'] = []
                    training_context['tmp_metrics']['D_fake_logit'].append(to_numpy(D_fake_logit).mean())  # adversarial_loss = torch.nn.BCEWithLogitsLoss()  # this_loss =
                    # adversarial_loss(self.D_fake - self.D_real.mean(0, keepdim=True),false_label+1)
                elif self.gan_type == 'ebgan':
                    pass

                training_context['current_loss'] = training_context['current_loss'] + this_loss
                if not self.gan_type == 'rasgan':
                    self.G_metric = self.D_fake
                    if 'D_fake' not in training_context['tmp_metrics']:
                        training_context['tmp_metrics']['D_fake'] = []
                        training_context['metrics']['D_fake'] = []
                    training_context['tmp_metrics']['D_fake'].append(to_numpy(self.D_fake).mean())

                if is_collect_data:
                    if 'gan_g_loss' not in training_context['losses']:
                        training_context['losses']['gan_g_loss'] = []
                    training_context['losses']['gan_g_loss'].append(float(to_numpy(this_loss)))
            except:
                PrintException()



        elif training_context['gan_role'] == 'discriminator':
            try:
                if self.generator_first == False:
                    training_context['retain_graph'] = True

                if self.use_total_variation:
                    self.D_real = self.D_real.clamp(min=-1, max=1)
                    self.D_fake = self.D_fake.clamp(min=-1, max=1)
                this_loss = 0
                if self.gan_type == 'gan':

                    adversarial_loss = BCELoss
                    real_loss = adversarial_loss(self.D_real, true_label)
                    fake_loss = adversarial_loss(self.D_fake, false_label)
                    this_loss = (real_loss + fake_loss).mean() / 2
                elif self.gan_type == 'dcgan':
                    adversarial_loss = torch.nn.BCELoss()
                    real_loss = adversarial_loss(self.D_real, true_label)
                    fake_loss = adversarial_loss(self.D_fake, false_label)
                    this_loss = (real_loss + fake_loss).mean() / 2
                elif self.gan_type == 'wgan':
                    this_loss = (self.D_fake - self.D_real).mean()
                elif self.gan_type == 'wgan-gp':
                    def compute_gradient_penalty():
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = to_tensor(np.random.random((self.img_real.size(0), 1, 1, 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = (alpha * self.img_real + ((1 - alpha) * self.img_fake)).requires_grad_(True)
                        out = self.discriminator(interpolates)
                        fake = to_tensor(np.ones(out.size()))
                        # Get gradient w.r.t. interpolates
                        gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=fake, create_graph=True,
                                                        retain_graph=True, only_inputs=True, )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    gp = 10 * compute_gradient_penalty()
                    if is_collect_data:
                        if 'gradient_penalty' not in training_context['losses']:
                            training_context['losses']['gradient_penalty'] = []
                        training_context['losses']['gradient_penalty'].append(float(to_numpy(gp)))

                    this_loss = gp + (self.D_fake - self.D_real).mean()
                elif self.gan_type == 'wgan-div':
                    k = 2
                    p = 6
                    # Compute W-div gradient penalty

                    real_grad = torch.autograd.grad(outputs=self.D_real, inputs=self.img_real, grad_outputs=true_label, create_graph=True, retain_graph=True, only_inputs=True, )[0]
                    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                    fake_grad = torch.autograd.grad(outputs=self.D_fake, inputs=self.img_fake, grad_outputs=true_label, create_graph=True, retain_graph=True, only_inputs=True, )[0]
                    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
                    if is_collect_data:
                        if 'gradient_penalty' not in training_context['losses']:
                            training_context['losses']['div_loss'] = []
                        training_context['losses']['div_loss'].append(float(to_numpy(div_gp)))
                    this_loss = (self.D_fake - self.D_real).mean() + div_gp
                elif self.gan_type == 'lsgan':
                    this_loss = 0.5 * (torch.mean((self.D_real - true_label) ** 2) + torch.mean(self.D_fake ** 2))
                elif self.gan_type == 'rasgan':
                    D_real_logit = sigmoid(self.D_real - to_tensor(to_numpy(self.D_fake.mean(0, True))))
                    D_fake_logit = sigmoid(self.D_fake - to_tensor(to_numpy(self.D_real.mean(0, True))))
                    this_loss = ((1 - D_real_logit) ** 2 + (0 - D_fake_logit) ** 2).mean() / 2
                    self.D_metric = D_real_logit
                    if 'D_real_logit' not in training_context['tmp_metrics']:
                        training_context['tmp_metrics']['D_real_logit'] = []
                        training_context['metrics']['D_real_logit'] = []
                    training_context['tmp_metrics']['D_real_logit'].append(to_numpy(
                        D_real_logit).mean())  # adversarial_loss = torch.nn.BCEWithLogitsLoss()  # this_loss =(
                    # adversarial_loss(self.D_real - self.D_fake.mean(0, keepdim=True),true_label)+ adversarial_loss(
                    # self.D_fake - self.D_real.mean(0, keepdim=True),false_label))/2.0

                training_context['current_loss'] = training_context['current_loss'] + this_loss
                if not self.gan_type == 'rasgan':
                    self.D_metric = self.D_real
                    if 'D_real' not in training_context['tmp_metrics']:
                        training_context['tmp_metrics']['D_real'] = []
                        training_context['metrics']['D_real'] = []
                    training_context['tmp_metrics']['D_real'].append(to_numpy(self.D_real).mean())

                if is_collect_data:
                    if 'gan_d_loss' not in training_context['losses']:
                        training_context['losses']['gan_d_loss'] = []
                    training_context['losses']['gan_d_loss'].append(float(to_numpy(this_loss)))
            except:
                PrintException()

    def on_optimization_step_end(self, training_context):
        model = training_context['current_model']
        is_collect_data = training_context['is_collect_data']

        if training_context['gan_role'] == 'generator':
            pass
            # self.img_fake = to_tensor(to_numpy(self.img_fake))


        elif training_context['gan_role'] == 'discriminator':
            if self.gan_type == 'wgan' or self.weight_clipping:
                for p in training_context['current_model'].parameters():
                    p.data.clamp_(-0.01, 0.01)

            # self.D_real = self.discriminator(self.img_real)  # self.D_fake = self.discriminator(self.img_fake)  #
            # training_context['D_real'] = self.D_real  # training_context['D_fake'] = self.D_fake  #
            # training_context['discriminator'] = model

    def on_batch_end(self, training_context):
        if training_context['gan_role'] == 'generator':
            if (training_context['current_epoch'] * training_context['total_batch'] + training_context[
                'current_batch'] + 1) % self.tile_image_frequency == 0:
                for i in range(3):
                    train_data = self.data_provider.next()
                    input = None
                    target = None
                    if 'signature' in training_context and len(training_context['signature']) > 0:
                        data_feed = training_context['signature']
                        input = to_tensor(train_data[data_feed.get('input')]) if data_feed.get('input') >= 0 else None
                        # target = to_tensor(train_data[signature.get('target')]) if signature.get('target') >= 0
                        # else None
                        imgs = to_numpy(self.generator(input)).transpose([0, 2, 3, 1]) * 127.5 + 127.5
                        self.tile_images.append(imgs)

                # if self.tile_image_include_mask:
                #     tile_images_list.append(input*127.5+127.5)
                tile_rgb_images(*self.tile_images, save_path=os.path.join('Results', 'tile_image_{0}.png'), imshow=True)
                self.tile_images = []
        if training_context['gan_role'] == 'generator' and self.noised_lr and training_context['current_epoch'] * \
                training_context['total_batch'] + training_context['current_batch'] + 1 > 1000:
            factor = math.cos(math.pi * (
                    training_context['current_epoch'] * training_context['total_batch'] + training_context[
                'current_batch'] + 1) / 100.0)
            base_lr = training_context['base_lr']
            training_context['optimizer'].adjust_learning_rate(
                base_lr * (1 + 0.2 * factor + 0.3 * (random.random() - 0.5)), False)
        if training_context['gan_role'] == 'discriminator' and training_context['current_epoch'] * training_context[
            'total_batch'] + training_context['current_batch'] + 1 == 12000:
            training_context['base_lr'] = training_context['base_lr'] / 2
        if training_context['gan_role'] == 'discriminator' and self.noised_lr and training_context['current_epoch'] * \
                training_context['total_batch'] + training_context['current_batch'] + 1 > 1000:
            factor = math.cos(math.pi * (
                    training_context['current_epoch'] * training_context['total_batch'] + training_context[
                'current_batch'] + 1) / 100.0)
            base_lr = training_context['base_lr']
            training_context['optimizer'].adjust_learning_rate(
                base_lr * (1 + 0.1 * factor + 0.1 * (random.random() - 0.5)), False)

    def on_epoch_end(self, training_context):
        try:
            if training_context['gan_role'] == 'discriminator':
                pass
                idx = 0 if self.generator_first else 1
                generator_metrics = self.training_items.value_list[idx].training_context['metrics'].value_list[0]
                discremnent_metrics = self.training_items.value_list[1 - idx].training_context['metrics'].value_list[0]
                clipping_range = 0.1

                if training_context['current_epoch'] > 5 and np.array(generator_metrics[-5:]).mean() < 0.25 and np.array(discremnent_metrics[-5:]).mean() > 0.75:
                    if 'clipping_range' not in training_context:
                        training_context['clipping_range'] = 0.3
                        self.g_train_frequency = 0.6

                    elif np.array(generator_metrics[-5:]).mean() < 0.2 or np.array(discremnent_metrics[-5:]).mean() > 0.8:
                        training_context['clipping_range'] = 0.1
                        self.g_train_frequency = 0.75

                    elif np.array(generator_metrics[-5:]).mean() < 0.15 or np.array(discremnent_metrics[-5:]).mean() > 0.85:
                        training_context['clipping_range'] = 0.05
                        self.g_train_frequency = 1

                    elif np.array(generator_metrics[-5:]).mean() < 0.1 or np.array(discremnent_metrics[-5:]).mean() > 0.9:
                        training_context['clipping_range'] = 0.01

                    for p in training_context['current_model'].parameters():
                        p.data.clamp_(-1 * training_context['clipping_range'], training_context['clipping_range'])

                # self.training_items.value_list[0].optimizer.lr / 2, True)  #
                # self.training_items.value_list[1].optimizer.adjust_learning_rate(  #
                # self.training_items.value_list[1].optimizer.lr / 2, True)  #         self.noise_end_epoch =
                # training_context['current_epoch'] + 10  #         self.experience_replay = False  #
                # self.noisy_labels = True

                # print(role1, grad1, metric1, role2, grad2, metric2)
        except:
            PrintException()

            #     if training_context['optimizer'].lr>1e-6:  #         training_context[
            #     'optimizer'].adjust_learning_rate(training_context['optimizer'].lr*0.5,True)  # elif
            #     training_context['current_epoch']>=1 and float(self.D_real.mean()) > 0.8 and float(
            #     self.D_fake.mean()) < 0.1 :  #     if self.discriminator is not None and model.name ==
            #     self.discriminator.name:  #         training_context['optimizer'].adjust_learning_rate(
            #     training_context['optimizer'].lr / 2.0)


class CycleGanCallback(GanCallbacksBase):
    # Generators: G_A: A -> B; G_B: B -> A.
    # Discriminators: D_A: G_B(B) vs. A   ; D_B: G_A(A) vs. B
    def __init__(self, generatorA=None, generatorB=None, discriminatorA=None, discriminatorB=None, gan_type='lsgan',
                 label_smoothing=False, noised_real=True, noise_intensity=0.05, weight_clipping=False,
                 tile_image_frequency=100, experience_replay=False, g_train_frequency=1, d_train_frequency=1,
                 cycle_loss_weight=10, identity_loss_weight=5, **kwargs):
        super(CycleGanCallback, self).__init__()
        if isinstance(generatorA, ImageGenerationModel):
            generatorA.model.name = 'generatorA'
            self.generatorA = generatorA.model
        if isinstance(generatorA, ImageGenerationModel):
            generatorB.model.name = 'generatorB'
            self.generatorB = generatorB.model
        if isinstance(discriminatorA, ImageClassificationModel):
            discriminatorA.model.name = 'discriminatorA'
            self.discriminatorA = discriminatorA.model
        if isinstance(discriminatorB, ImageClassificationModel):
            discriminatorB.model.name = 'discriminatorB'
            self.discriminatorB = discriminatorB.model
        self.data_provider = None

        self.D_realA = None
        self.D_fakeA = None
        self.D_realB = None
        self.D_fakeB = None
        self.D_metric = None
        self.G_metric = None
        self.realA = None
        self.realB = None
        self.fakeA = None  # B->A
        self.fakeB = None  # A->B
        self.fakeA_buffer = ReplayBuffer(1000)
        self.fakeB_buffer = ReplayBuffer(1000)
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.gan_type = gan_type
        self.label_smoothing = label_smoothing
        self.noised_real = noised_real
        self.noise_intensity = noise_intensity
        self.tile_image_frequency = tile_image_frequency
        self.weight_clipping = weight_clipping
        self.experience_replay = experience_replay
        self.g_train_frequency = g_train_frequency
        self.d_train_frequency = d_train_frequency
        if self.experience_replay == True:
            make_dir_if_need('Replay')
        self.tile_images = []
        self.generator_first = None
        self.cooldown_counter = 0
        self.beginning_repository = []
        self.latter_repository = []

    def on_training_start(self, training_context):
        training_items = training_context['training_items']
        self.data_provider = training_context['_dataloaders'].value_list[0]

        conterparty = OrderedDict()
        conterparty['generatorA'] = None
        conterparty['generatorB'] = None
        conterparty['discriminatorA'] = None
        conterparty['discriminatorB'] = None

        data_feed = OrderedDict()
        data_feed['input'] = 0
        data_feed['target'] = 1

        for k, training_item in training_items.items():
            if isinstance(training_item, ImageGenerationModel):
                if self.generatorA.name == training_item.model.name:
                    conterparty['generatorA'] = training_item
                    training_item.training_context['gan_role'] = 'generatorA'
                    training_item.training_context['signature'] = data_feed
                elif self.generatorB.name == training_item.model.name:
                    conterparty['generatorB'] = training_item
                    training_item.training_context['gan_role'] = 'generatorB'
                elif self.generatorA is None and self.generatorB is None:
                    self.generatorA = training_item.model
                    conterparty['generatorA'] = training_item
                    training_item.training_context['gan_role'] = 'generatorA'
                elif self.generatorA is not None and self.generatorB is None:
                    self.generatorB = training_item.model
                    conterparty['generatorB'] = training_item
                    training_item.training_context['gan_role'] = 'generatorB'

            elif isinstance(training_item, ImageClassificationModel):
                if self.discriminatorA.name == training_item.model.name:
                    conterparty['discriminatorA'] = training_item
                    training_item.training_context['gan_role'] = 'discriminatorA'
                elif self.discriminatorB.name == training_item.model.name:
                    conterparty['discriminatorB'] = training_item
                    training_item.training_context['gan_role'] = 'discriminatorB'
                elif self.discriminatorA is None and self.discriminatorB is None:
                    self.discriminatorA = training_item.model
                    conterparty['discriminatorA'] = training_item
                    training_item.training_context['gan_role'] = 'discriminatorA'
                elif self.discriminatorA is not None and self.discriminatorB is None:
                    conterparty['discriminatorB'] = training_item
                    self.discriminatorB = training_item.model
                    training_item.training_context['gan_role'] = 'discriminatorB'

        if self.generatorA is not None and self.generatorB is None:
            self.generatorB = self.generatorA.copy()
            self.generatorB.training_context['gan_role'] = 'generatorB'
            conterparty['generatorB'] = self.generatorB
        if self.discriminatorA is not None and self.discriminatorB is None:
            self.discriminatorB = self.discriminatorA.copy()
            self.discriminatorB.training_context['gan_role'] = 'discriminatorB'
            conterparty['discriminatorB'] = self.discriminatorB

        conterparty['generatorA'].optimizer.param_groups[0]['params'] = itertools.chain(
            self.generatorA.trainable_weights, self.generatorB.trainable_weights)
        # conterparty['generatorB'].optimizer.param_groups[0]['params'] = itertools.chain(
        # self.generatorA.trainable_weights, self.generatorB.trainable_weights)
        conterparty['generatorB'].training_context['stop_update'] = sys.maxsize
        training_context['training_items'] = conterparty
        self.generator_first = True

    def on_data_received(self, training_context):
        curr_epochs = training_context['current_epoch']
        tot_epochs = training_context['total_epoch']
        try:
            if training_context['gan_role'] == 'generatorA':
                self.realA = training_context['current_input']
                self.realB = training_context['current_target']
                self.fakeA = self.generatorB(self.realB)
                self.fakeB = self.generatorA(self.realA)

            if training_context['gan_role'] == 'generatorA' or training_context['gan_role'] == 'generatorB':
                self.D_realA = self.discriminatorA(self.realA)
                self.D_fakeA = self.discriminatorA(self.fakeA)
                self.D_realB = self.discriminatorB(self.realB)
                self.D_fakeB = self.discriminatorB(self.fakeB)

            elif training_context['gan_role'] == 'discriminatorA' or training_context['gan_role'] == 'discriminatorB':
                fakeA = self.fakeA
                if self.experience_replay:
                    fakeA = self.fakeA_buffer.push_and_pop(self.fakeA)
                realA = self.realA
                if self.noised_real:
                    realA = (self.realA + to_tensor(
                        self.noise_intensity * (1 - float(curr_epochs) / (tot_epochs)) * np.random.standard_normal(
                            list(self.realA.size())))).clamp(-1, 1)

                self.D_realA = self.discriminatorA(realA)
                self.D_fakeA = self.discriminatorA(fakeA)

            elif training_context['gan_role'] == 'discriminatorA' or training_context['gan_role'] == 'discriminatorB':
                fakeB = self.fakeB
                if self.experience_replay:
                    fakeB = self.fakeB_buffer.push_and_pop(self.fakeB)
                realB = self.realB
                if self.noised_real:
                    realB = (self.realB + to_tensor(
                        self.noise_intensity * (1 - float(curr_epochs) / (tot_epochs)) * np.random.standard_normal(
                            list(self.realB.size())))).clamp(-1, 1)

                self.D_realB = self.discriminatorB(realB)
                self.D_fakeB = self.discriminatorB(fakeB)

        except:
            PrintException()

    def on_loss_calculation_end(self, training_context):
        model = training_context['current_model']
        current_mode = None
        is_collect_data = training_context['is_collect_data']

        true_label = to_tensor(np.ones((self.D_realA.size()), dtype=np.float32))
        false_label = to_tensor(np.zeros((self.D_realA.size()), dtype=np.float32))

        if self.label_smoothing:
            if training_context['current_epoch'] < 20:
                true_label = to_tensor(np.random.randint(80, 100, (self.D_realA.size())).astype(np.float32) / 100.0)
            elif training_context['current_epoch'] < 50:
                true_label = to_tensor(np.random.randint(85, 100, (self.D_realA.size())).astype(np.float32) / 100.0)
            elif training_context['current_epoch'] < 200:
                true_label = to_tensor(np.random.randint(90, 100, (self.D_realA.size())).astype(np.float32) / 100.0)
            else:
                pass
        # true_label.requires_grad=False
        # false_label.requires_grad=False

        if training_context['gan_role'] == ['generatorA', 'generatorB']:
            try:

                this_lossA = 0
                this_lossB = 0

                if self.gan_type == 'gan':
                    adversarial_loss = torch.nn.BCELoss()
                    this_lossA = adversarial_loss(self.D_fakeA, true_label).mean()
                    this_lossB = adversarial_loss(self.D_fakeB, true_label).mean()

                elif self.gan_type in ['wgan', 'wgan-gp']:
                    this_lossA = -self.D_fakeA.mean()
                    this_lossB = -self.D_fakeB.mean()

                elif self.gan_type == 'lsgan':
                    this_lossA = torch.mean((self.D_fakeA - 1) ** 2)
                    this_lossB = torch.mean((self.D_fakeB - 1) ** 2)
                loss_gan = (this_lossA + this_lossB) / 2
                training_context['current_loss'] = training_context['current_loss'] + loss_gan

                loss_id_A = L1Loss()(self.generatorB(self.realA), self.realA)
                loss_id_B = L1Loss()(self.generatorA(self.realB), self.realB)
                loss_identity = (loss_id_A + loss_id_B).mean() / 2
                training_context['current_loss'] = training_context[
                                                       'current_loss'] + self.identity_loss_weight * loss_identity

                # Cycle loss
                recovA = self.generatorB(self.fakeB)
                loss_cycle_A = L1Loss()(recovA, self.realA)
                recovB = self.generatorA(self.fakeA)
                loss_cycle_B = L1Loss()(recovB, self.realB)

                loss_cycle = (loss_cycle_A + loss_cycle_B).mean() / 2
                training_context['current_loss'] = training_context[
                                                       'current_loss'] + self.cycle_loss_weight * loss_cycle

                self.G_metric = self.D_fakeA
                if 'D_fakeA' not in training_context['tmp_metrics']:
                    training_context['tmp_metrics']['D_fakeA'] = []
                    training_context['metrics']['D_fakeA'] = []
                training_context['tmp_metrics']['D_fakeA'].append(to_numpy(self.D_fakeA).mean())

                if is_collect_data:
                    if 'gan_ga_loss' not in training_context['losses']:
                        training_context['losses']['gan_ga_loss'] = []
                    if 'gan_gb_loss' not in training_context['losses']:
                        training_context['losses']['gan_gb_loss'] = []
                    if 'identity_a_loss' not in training_context['losses']:
                        training_context['losses']['identity_a_loss'] = []
                    if 'identity_b_loss' not in training_context['losses']:
                        training_context['losses']['identity_b_loss'] = []
                    if 'cycle_a_loss' not in training_context['losses']:
                        training_context['losses']['cycle_a_loss'] = []
                    if 'cycle_b_loss' not in training_context['losses']:
                        training_context['losses']['cycle_b_loss'] = []
                    training_context['losses']['gan_ga_loss'].append(float(to_numpy(this_lossA)))
                    training_context['losses']['gan_gb_loss'].append(float(to_numpy(this_lossB)))
                    training_context['losses']['identity_a_loss'].append(float(to_numpy(loss_id_A)))
                    training_context['losses']['identity_b_loss'].append(float(to_numpy(loss_id_B)))
                    training_context['losses']['cycle_a_loss'].append(float(to_numpy(loss_cycle_A)))
                    training_context['losses']['cycle_b_loss'].append(float(to_numpy(loss_cycle_B)))
            except:
                PrintException()



        elif training_context['gan_role'] == 'discriminatorA':
            try:

                this_loss = 0
                if self.gan_type == 'vanilla':
                    adversarial_loss = torch.nn.BCEWithLogitsLoss()
                    real_loss = adversarial_loss(self.D_realA, true_label)
                    fake_loss = adversarial_loss(self.D_fakeA, false_label)
                    this_loss = (real_loss + fake_loss).mean() / 2
                elif self.gan_type == 'wgan':
                    this_loss = -self.D_realA.mean() + self.D_fakeA.mean()
                elif self.gan_type == 'wgan-gp':
                    def compute_gradient_penalty():
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = to_tensor(np.random.random((self.realA.size(0), 1, 1, 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = (alpha * self.realA + ((1 - alpha) * self.fakeA)).requires_grad_(True)
                        out = self.discriminator(interpolates)
                        fake = to_tensor(np.ones(out.size()))
                        # Get gradient w.r.t. interpolates
                        gradients = \
                            torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=fake, create_graph=True,
                                                retain_graph=True, only_inputs=True, )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty

                    this_loss = 10 * compute_gradient_penalty() - self.D_realA.mean() + self.D_fakeA.mean()

                elif self.gan_type == 'lsgan':
                    this_loss = 0.5 * (torch.mean((self.D_realA - true_label) ** 2) + torch.mean(self.D_fakeA ** 2))

                training_context['current_loss'] = training_context['current_loss'] + this_loss

                self.D_metric = self.D_realA
                if 'D_realA' not in training_context['tmp_metrics']:
                    training_context['tmp_metrics']['D_realA'] = []
                    training_context['metrics']['D_realA'] = []
                training_context['tmp_metrics']['D_realA'].append(to_numpy(self.D_realA).mean())

                if is_collect_data:
                    if 'gan_da_loss' not in training_context['losses']:
                        training_context['losses']['gan_da_loss'] = []
                    training_context['losses']['gan_da_loss'].append(float(to_numpy(this_loss)))
            except:
                PrintException()
        elif training_context['gan_role'] == 'discriminatorB':
            try:
                this_loss = 0
                if self.gan_type == 'vanilla':
                    adversarial_loss = torch.nn.BCEWithLogitsLoss()
                    real_loss = adversarial_loss(self.D_realB, true_label)
                    fake_loss = adversarial_loss(self.D_fakeB, false_label)
                    this_loss = (real_loss + fake_loss).mean() / 2
                elif self.gan_type == 'wgan':
                    this_loss = -self.D_realB.mean() + self.D_fakeB.mean()
                elif self.gan_type == 'wgan-gp':
                    def compute_gradient_penalty():
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = to_tensor(np.random.random((self.realB.size(0), 1, 1, 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = (alpha * self.realB + ((1 - alpha) * self.fakeB)).requires_grad_(True)
                        out = self.discriminator(interpolates)
                        fake = to_tensor(np.ones(out.size()))
                        # Get gradient w.r.t. interpolates
                        gradients = \
                            torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=fake, create_graph=True,
                                                retain_graph=True, only_inputs=True, )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty

                    this_loss = 10 * compute_gradient_penalty() - self.D_realB.mean() + self.D_fakeB.mean()

                elif self.gan_type == 'lsgan':
                    this_loss = 0.5 * (torch.mean((self.D_realB - true_label) ** 2) + torch.mean(self.D_fakeB ** 2))

                training_context['current_loss'] = training_context['current_loss'] + this_loss

                self.D_metric = self.D_realB
                if 'D_realB' not in training_context['tmp_metrics']:
                    training_context['tmp_metrics']['D_realB'] = []
                    training_context['metrics']['D_realB'] = []
                training_context['tmp_metrics']['D_realB'].append(to_numpy(self.D_realB).mean())

                if is_collect_data:
                    if 'gan_db_loss' not in training_context['losses']:
                        training_context['losses']['gan_db_loss'] = []
                    training_context['losses']['gan_db_loss'].append(float(to_numpy(this_loss)))

            except:
                PrintException()

    def on_optimization_step_end(self, training_context):
        model = training_context['current_model']
        is_collect_data = training_context['is_collect_data']

        if training_context['gan_role'] == ['generatorA', 'generatorB']:
            pass

            # training_context['img_fake'] = self.img_fake  # self.D_fake = self.discriminator(self.img_fake)  #
            # training_context['D_fake'] = self.D_fake  #  # if self.gan_type == 'gan':  #     adversarial_loss =
            # torch.nn.BCELoss()  #     this_loss = adversarial_loss(self.D_fake, true_label)  # elif self.gan_type
            # == 'wgan':  #     this_loss = -self.D_fake.mean()


        elif training_context['gan_role'] == 'discriminatorA':
            if self.experience_replay and (training_context['current_batch'] + 1) % 100 == 0:
                np.save('Replay/fakeA_buffer.npy', self.fakeA_buffer)

            if self.gan_type == 'wgan' or self.weight_clipping:
                for p in training_context['current_model'].parameters():
                    p.data.clamp_(-0.01, 0.01)

        elif training_context['gan_role'] == 'discriminatorB':
            if self.experience_replay and (training_context['current_batch'] + 1) % 100 == 0:
                np.save('Replay/fakeB_buffer.npy', self.fakeB_buffer)

            if self.gan_type == 'wgan' or self.weight_clipping:
                for p in training_context['current_model'].parameters():
                    p.data.clamp_(-0.01, 0.01)

            # self.D_real = self.discriminator(self.img_real)  # self.D_fake = self.discriminator(self.img_fake)  #
            # training_context['D_real'] = self.D_real  # training_context['D_fake'] = self.D_fake  #
            # training_context['discriminator'] = model

    def on_batch_end(self, training_context):
        model = training_context['current_model']
        if training_context['gan_role'] == 'generatorA':
            if (training_context['current_epoch'] * training_context['total_batch'] + training_context[
                'current_batch'] + 1) % self.tile_image_frequency == 0:
                self.tile_images = []
                self.tile_images.append(to_numpy(self.realA).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                self.tile_images.append(to_numpy(self.fakeA).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                self.tile_images.append(to_numpy(self.realB).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                self.tile_images.append(to_numpy(self.fakeB).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                tile_rgb_images(*self.tile_images, save_path=os.path.join('Results', 'tile_image_{0}.png'), imshow=True)
                self.tile_images = []

    def on_epoch_end(self, training_context):

        if (self.generator_first and training_context['gan_role'] == 'discriminator') or (
                not self.generator_first and training_context['gan_role'] == 'generator'):
            if (training_context['current_epoch'] + 1) % 10 == 0:
                if training_context['optimizer'].lr > 1e-6:
                    if np.array(training_context['grads_state']['last_layer'][-10:]).mean() < 2e-3 and np.array(
                            training_context['grads_state']['last_layer'][-10:]).mean() < 2e-3:
                        training_context['optimizer'].adjust_learning_rate(training_context['optimizer'].lr * 0.5,
                                                                           True)  # elif training_context[
                        # 'current_epoch']>=1 and float(self.D_real.mean()) > 0.8 and float(self.D_fake.mean()) < 0.1
                        # :  #     if self.discriminator is not None and model.name == self.discriminator.name:  #         training_context['optimizer'].adjust_learning_rate(training_context['optimizer'].lr / 2.0)









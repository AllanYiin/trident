from configparser import ConfigParser

import pytest
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from test_legacy_training import make_plan
from trident.backend.pytorch_ops import add, binary_cross_entropy, moments, to_tensor
from trident.optims.pytorch_losses import CrossEntropyLoss, MSELoss, StyleLoss
from trident.optims.pytorch_optimizers import Adam, Apollo, GaLore, Lion, Ranger21, RangerAdaBelief
from trident.optims.pytorch_regularizers import l1_reg, l2_reg
from trident.callbacks import ConstantWarmup, CosineLR, CosineWarmup, ExponentialWarmup, LinearWarmup


@pytest.mark.parametrize('policy, expected', [
    (LinearWarmup, [.0625, .296875, .53125, .765625, 1]),
    (CosineWarmup, [.0625, .1997936963, .53125, .8627063037, 1]),
    (ExponentialWarmup, [.0625, .125, .25, .5, 1]),
    (ConstantWarmup, [.0625] * 5),
])
def test_warmup_actual_group_rates_with_accumulation_and_cosine(tmp_path, policy, expected):
    plan, item = make_plan(tmp_path, rows=16, batch_size=1, accumulation=2)
    extra = nn.Parameter(torch.ones_like(item.model.weight))
    item.optimizer.add_param_group(dict(params=[extra], lr=.01))
    warmup = policy.from_ratio(.625, total_updates=8, start_factor=.0625)
    item.with_learning_rate_scheduler(CosineLR(period=2, min_lr=0), warmup=warmup)
    rates = []
    item.optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: rates.append([g['lr'] for g in opt.param_groups]))
    plan.start_now()
    np.testing.assert_allclose(rates, np.outer(expected + [1, 1, .5], [.001, .01]))


def test_ini_cosine_warmup_matches_explicit_policy_through_handoff(tmp_path):
    ini = ConfigParser()
    ini.read_string('[training]\nwarmup="cosine"')
    explicit = CosineWarmup.from_ratio(.03, total_updates=12_000)
    _, short = make_plan(tmp_path)
    _, long = make_plan(tmp_path)
    for item, warmup in [(short, ini['training']['warmup']), (long, explicit)]:
        item.with_learning_rate_scheduler(CosineLR(period=2, min_lr=0), warmup=warmup)
    assert short.warmup == long.warmup == explicit
    assert short.lr_scheduler.config_dict() == long.lr_scheduler.config_dict()
    with pytest.raises(ValueError, match="unknown warmup preset"):
        short.with_learning_rate_scheduler(warmup="cosnie")

    rates = []
    for index in range(364):
        for item in (short, long):
            item.training_context.update(current_epoch=0, total_epoch=1, total_batch=364,
                                         current_batch=index, steps=index, current_loss=item.model.weight.sum())
            item.do_gradient_update()
        assert short.current_lr == long.current_lr
        rates.append(short.current_lr)
        if index == 179:
            short.lr_scheduler.load_state_dict(long.lr_scheduler.state_dict())
    assert rates[0] == pytest.approx(.001 * .01)
    assert rates[359] == pytest.approx(.001)
    assert rates[-1] < rates[359]


def test_warmup_ratio_rounding_and_validation():
    assert CosineWarmup.from_ratio(.07, total_updates=100).steps == 7
    assert ConstantWarmup.from_ratio(.1, total_updates=13).steps == 2
    assert ConstantWarmup.from_ratio(.01, total_updates=2).factor_at(0) == .01
    for ratio in (0, -1, True, float('nan'), 1.01):
        with pytest.raises(ValueError, match='ratio'):
            LinearWarmup.from_ratio(ratio, total_updates=10)
    with pytest.raises(ValueError, match='total_updates'):
        LinearWarmup.from_ratio(.1, total_updates=1.5)
    with pytest.raises(ValueError, match='positive start_factor'):
        ExponentialWarmup(5, 0)


def test_warmup_checkpoint_distinguishes_curves_and_loads_legacy_linear(tmp_path):
    _, item = make_plan(tmp_path)
    item.with_learning_rate_scheduler(warmup=LinearWarmup(4))
    state = item.lr_scheduler.state_dict()
    assert state['config']['warmup'] == {'steps': 4, 'start_factor': .01}
    item.lr_scheduler.load_state_dict(state)
    item.with_learning_rate_scheduler(warmup=CosineWarmup(4))
    with pytest.raises(ValueError, match='configuration'):
        item.lr_scheduler.load_state_dict(state)


def test_epoch_warmup_without_main_scheduler(tmp_path):
    plan, item = make_plan(tmp_path, rows=1, batch_size=1)
    item.with_learning_rate_scheduler(warmup=2)
    rates = []
    item.optimizer.register_step_pre_hook(lambda opt, args, kwargs: rates.append(opt.param_groups[0]['lr']))
    plan.repeat_epochs(3).start_now()
    assert rates == pytest.approx([.00001, .000505, .001])


def test_single_step_warmup_preserves_tensor_lr_storage(tmp_path):
    plan, item = make_plan(tmp_path, rows=2, batch_size=1)
    lr = torch.tensor(.001)
    item.optimizer.param_groups[0]['lr'] = lr
    item.with_learning_rate_scheduler(warmup=LinearWarmup(1, 0))
    rates = []
    item.optimizer.register_step_pre_hook(lambda opt, args, kwargs: rates.append(float(opt.param_groups[0]['lr'])))
    plan.start_now()
    assert item.optimizer.param_groups[0]['lr'] is lr
    assert rates == pytest.approx([.001, .001])


def test_one_cycle_uses_remaining_updates_and_keeps_short_tail(tmp_path):
    from trident.callbacks.lr_schedulers import OnceCycleLR
    plan, item = make_plan(tmp_path, rows=5, batch_size=1, accumulation=2)
    main = OnceCycleLR(lr_range=(.0001, .001))
    item.with_learning_rate_scheduler(main, warmup=LinearWarmup(2))
    plan.repeat_epochs(2).start_now()
    assert main.num_steps == item.lr_scheduler.main_updates == 4
    assert item.lr_scheduler.updates == 6


@pytest.mark.parametrize('checkpoint_after', [1, 3, 6])
def test_callback_scheduler_checkpoint_replays_group_lrs(tmp_path, checkpoint_after):
    def make_item():
        _, item = make_plan(tmp_path)
        item.with_learning_rate_scheduler('CosineLR', period=2, min_lr=0, warmup=LinearWarmup(3, .2))
        return item

    def update(item, i):
        item.training_context.update(optimizer=item.optimizer, steps=i, current_batch=i,
                                     current_epoch=0, total_epoch=1, total_batch=10)
        item.lr_scheduler.on_optimization_step_start(item.training_context)
        rate = item.optimizer.param_groups[0]['lr']
        item.optimizer.step()
        item.lr_scheduler.on_optimization_step_end(item.training_context)
        return rate

    original = make_item()
    for i in range(checkpoint_after):
        update(original, i)
    checkpoint = tmp_path / 'schedule.pt'
    torch.save(dict(optimizer=original.optimizer.state_dict(), scheduler=original.lr_scheduler.state_dict()), checkpoint)
    resumed = make_item()
    state = torch.load(checkpoint, weights_only=True)
    resumed.optimizer.load_state_dict(state['optimizer'])
    resumed.lr_scheduler.load_state_dict(state['scheduler'])
    for i in range(checkpoint_after, 10):
        assert update(resumed, i) == update(original, i)


def test_external_and_optimizer_warmup_conflict(tmp_path):
    from trident.optims.pytorch_optimizers import AdamW
    _, item = make_plan(tmp_path)
    item.with_optimizer(AdamW, lr=.01, warmup=10)
    with pytest.raises(ValueError, match='built-in warmup'):
        item.with_learning_rate_scheduler(warmup=LinearWarmup(3))


@pytest.mark.parametrize('after_first_update', [False, True])
def test_optimizer_group_added_after_setup_cannot_enable_second_warmup(tmp_path, after_first_update):
    from trident.optims.pytorch_optimizers import AdamW

    _, item = make_plan(tmp_path)
    item.with_optimizer(AdamW, lr=.01).with_learning_rate_scheduler(warmup=CosineWarmup(3))
    weight = item.model.weight
    if after_first_update:
        item.training_context.update(current_loss=weight.sum(), current_batch=0, steps=0)
        item.do_gradient_update()
    updates = item.lr_scheduler.updates
    rates = [group['lr'] for group in item.optimizer.param_groups]
    extra = nn.Parameter(torch.ones_like(weight))
    item.optimizer.add_param_group({'params': [extra], 'lr': .02, 'warmup': 10})
    item.training_context.update(current_loss=(weight + extra).sum(), current_batch=1, steps=1)
    with pytest.raises(ValueError, match='built-in warmup'):
        item.do_gradient_update()
    assert item.lr_scheduler.updates == updates
    assert [group['lr'] for group in item.optimizer.param_groups] == rates + [.02]


def test_regularizers_and_tensor_ops_preserve_the_graph():
    weight = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        weight.weight.fill_(2)
    for reg, value in [(l1_reg, 1.), (l2_reg, 4.)]:
        (gradient,) = torch.autograd.grad(reg(weight, 1.), weight.weight)
        torch.testing.assert_close(gradient, torch.full_like(weight.weight, value))

    x = torch.tensor([1., 3.], requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad(add(np.ones(2, np.float32), x).sum(), x)[0],
                               torch.ones(2))
    mean, variance = moments(x, 0, keepdims=False)
    torch.testing.assert_close(torch.autograd.grad(mean + variance, x)[0], torch.tensor([-.5, 1.5]))
    assert to_tensor(x, dtype=torch.float64, device='cpu').dtype == torch.float64
    assert not to_tensor(x * 2, requires_grad=False, device='cpu').requires_grad
    assert x.requires_grad


def test_logits_losses_match_torch_including_soft_labels_and_ignore():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logits = torch.tensor([-3., 3.], device=device, requires_grad=True)
    targets = logits.new_tensor([1., 0.])
    ours = binary_cross_entropy(logits, targets, from_logits=True).sum()
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    torch.testing.assert_close(torch.autograd.grad(ours, logits)[0], torch.autograd.grad(expected, logits)[0])

    for logits, target in (
        ([[.9, .1, 0.]], [[.7, .3, 0.]]),
        ([[2., 0., -1.], [1., 2., 3.]], [0, -1]),
    ):
        logits = torch.tensor(logits, device=device, requires_grad=True)
        target = torch.tensor(target, device=device)
        own = CrossEntropyLoss(from_logits=True, ignore_index=-1)(logits, target)
        expected = F.cross_entropy(logits, target, ignore_index=-1)
        torch.testing.assert_close(torch.autograd.grad(own, logits)[0], torch.autograd.grad(expected, logits)[0])


def test_batch_sum_and_weighted_classification_accumulate(tmp_path):
    reference, full = make_plan(tmp_path, rows=4, batch_size=4)
    micro, accumulated = make_plan(tmp_path, rows=4, batch_size=2, accumulation=2)
    for item in (full, accumulated):
        item._losses.clear()
        item.with_loss(MSELoss(reduction='batch_sum'))
    reference.start_now()
    micro.start_now()
    torch.testing.assert_close(full.model.weight, accumulated.model.weight)

    # One weighted mean with a different denominator in each microbatch.
    _, full = make_plan(tmp_path, rows=4, batch_size=4)
    _, accumulated = make_plan(tmp_path, rows=4, batch_size=2, accumulation=2)
    for item in (full, accumulated):
        item._losses.clear()
        weight = item.model.weight
        ce = CrossEntropyLoss(from_logits=True, ignore_index=-1, sample_weight=[1., 3.])
        features = weight.new_tensor([.5, 2., 1., 3.])
        labels = torch.tensor([0, -1, 1, 1], device=weight.device)
        batch_size = 4 if item.accumulation_steps == 1 else 2
        for offset in range(0, 4, batch_size):
            z = features[offset:offset + batch_size] * weight.reshape(())
            loss = ce(torch.stack((z, -z), dim=1), labels[offset:offset + batch_size])
            item.training_context.update(
                current_loss=loss, current_batch=offset // batch_size, total_batch=4 // batch_size,
                normalized_losses={'CE': (loss, ce.gradient_denominator)},
                train_data={'x': z}, steps=offset // batch_size,
            )
            item.do_gradient_update()
    torch.testing.assert_close(full.model.weight, accumulated.model.weight)


def test_only_managed_gradients_are_accumulated(tmp_path):
    _, item = make_plan(tmp_path)
    outsider = nn.Parameter(torch.ones_like(item.model.weight))
    item.training_context.update(current_loss=(item.model.weight * outsider).sum(), current_batch=0, steps=0)
    item.do_gradient_update()
    assert outsider.grad is None

    target = torch.ones(1, 2, 2, 2, requires_grad=True)
    feature = torch.full_like(target, 2, requires_grad=True)
    StyleLoss()(feature, target).backward()
    assert target.grad is None
    assert feature.grad is not None


def test_delayed_items_start_and_disconnected_loss_cannot_step(tmp_path, monkeypatch):
    plan, item = make_plan(tmp_path, rows=2, batch_size=2)
    item.start_epoch = 1
    plan.repeat_epochs(2).start_now()
    assert item.model.weight.item() > 1

    monkeypatch.setattr(item.optimizer, 'step', lambda: pytest.fail('disconnected loss called step'))
    item.training_context.update(current_loss=torch.tensor(0., requires_grad=True), current_batch=0)
    item.do_gradient_update()


@pytest.mark.parametrize("optimizer", [Adam, Apollo, GaLore, Lion, RangerAdaBelief])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_optimizer_update(optimizer, dtype):
    p = nn.Parameter(torch.ones((4, 3), dtype=dtype))
    optimizer = optimizer([p], lr=.03)
    p.grad = torch.ones_like(p)
    optimizer.step()
    assert p.isfinite().all()
    assert not torch.equal(p, torch.ones_like(p))


def test_ranger21_and_adam_invalid_gradient_do_not_corrupt_parameters():
    p = nn.Parameter(torch.ones(2, 3))
    optimizer = Ranger21([p], lr=.03)
    p.grad = torch.ones_like(p)
    optimizer.step()
    assert p.isfinite().all()

    opt = Adam([p], lr=.1)
    opt.step()
    expected = p.detach().clone()
    p.grad = torch.full_like(p, float('nan'))
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        opt.step()
    assert opt.state[p]['step'] == 1
    torch.testing.assert_close(p, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP coverage')
def test_amp_overflow_does_not_report_a_step(tmp_path, monkeypatch):
    class Overflow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.shape = value.shape
            return value.new_ones(())
        @staticmethod
        def backward(ctx, grad):
            return grad.new_full(ctx.shape, float('inf'))

    _, item = make_plan(tmp_path)
    item.with_automatic_mixed_precision_training()
    item.with_learning_rate_scheduler(warmup=LinearWarmup(3))
    item.gradscaler = torch.amp.GradScaler('cuda', init_scale=8)
    expected = item.model.weight.detach().clone()
    monkeypatch.setattr(type(item), 'do_on_optimization_step_end',
                        lambda self: pytest.fail('overflow reported a completed step'))
    item.training_context.update(current_loss=Overflow.apply(item.model.weight), current_batch=0, steps=0)
    item.do_gradient_update()
    assert item.gradscaler.get_scale() == 4
    assert item.lr_scheduler.updates == 0
    torch.testing.assert_close(item.model.weight, expected)



def test_center_loss_parameters_are_managed_and_updated(tmp_path):
    from trident.optims.pytorch_losses import CenterLoss

    _, item = make_plan(tmp_path)
    loss = CenterLoss(num_classes=2, feat_dim=1)
    item.with_loss(loss)
    assert any(p is loss.centers for group in item.optimizer.param_groups for p in group['params'])
    feature = item.model.weight.new_tensor([[1.5], [2.0]])
    labels = torch.tensor([0, 1], device=feature.device)
    old = loss.centers.detach().clone()
    item.training_context.update(current_loss=loss(feature, labels), current_batch=0, steps=0)
    item.do_gradient_update()
    assert not torch.equal(loss.centers, old)


def test_gradient_penalty_norm_covers_all_channels():
    from trident.optims.trainers import GanTrainingPlan

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.4))

        @property
        def model(self):
            return self

        def forward(self, inputs):
            return self.weight * inputs.flatten(1).sum(1)

        def with_loss(self, loss, loss_weight):
            self.penalty = loss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    critic = Critic().to(device)
    plan = GanTrainingPlan(None, critic)
    plan.with_gradient_penalty()
    real = torch.ones((2, 3, 2, 2), device=device, requires_grad=True)
    fake = real * 2
    penalty = critic.penalty(real, fake)
    expected = (critic.weight * (12 ** .5) - 1).square()
    torch.testing.assert_close(penalty, expected)
    (grad,) = torch.autograd.grad(penalty, critic.weight)
    (reference,) = torch.autograd.grad(expected, critic.weight)
    torch.testing.assert_close(grad, reference)

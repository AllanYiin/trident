import copy
import datetime
import os
from pathlib import Path

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import pytest
import torch
import torch.distributed as dist

from trident.data.pipeline import ByteTokenizer, CausalLMCollator, Dataset, Iterator
from trident.optims.language import (CausalLM, CausalLMConfig, CausalLanguageModel, CausalLMObjective,
                                    ObjectiveResult, GRPO, PPO, RLTrainingPlan, ValueModel)
from trident.optims.trainers import TrainingPlan


def tiny():
    return CausalLM(CausalLMConfig(vocab_size=272, hidden_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=1,
                                   intermediate_size=32, max_position_embeddings=64))


def data(batch_size=1):
    return Iterator(Dataset([{"text": value} for value in ("abc", "abcdefgh", "a", "xyz")]),
                    batch_size=batch_size, collate=CausalLMCollator(ByteTokenizer(), max_length=32))


def _ddp_worker(rank, directory):
    torch.set_num_threads(1)
    torch.manual_seed(51)
    directory = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(directory / "rendezvous").as_uri(),
                            timeout=datetime.timedelta(seconds=60))
    try:
        item = (CausalLanguageModel(tiny(), ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.01)
                .with_accumulate_grads(2))
        plan = (TrainingPlan().add_training_item(item).with_data_loader(data())
                .with_distributed(backend="gloo").with_checkpointing(directory / "checkpoint", every_updates=1))
        plan.start_now()
        torch.save(item.model.module.state_dict(), directory / ("rank{}.pt".format(rank)))
    finally:
        dist.destroy_process_group()


def _ddp_rl_worker(rank, directory, algorithm):
    torch.set_num_threads(1)
    torch.manual_seed(51 + rank)
    directory = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(directory / "rendezvous").as_uri(),
                            timeout=datetime.timedelta(seconds=60))
    try:
        model = tiny()
        reference = copy.deepcopy(model)
        objective = PPO(reference, ValueModel(copy.deepcopy(model))) if algorithm == "ppo" else GRPO(reference)
        item = (CausalLanguageModel(model, ByteTokenizer()).with_objective(objective)
                .with_optimizer(torch.optim.AdamW, lr=0.001).with_accumulate_grads(2))
        loader = Iterator(Dataset([{"prompt": "Q"}, {"prompt": "R"}]), collate=list)
        plan = (RLTrainingPlan().add_training_item(item).with_data_loader(loader).with_distributed(backend="gloo")
                .with_rollout(num_generations=4, max_new_tokens=3).with_policy_updates(batch_size=1)
                .with_reward(lambda prompt, completion, sample: float(sum(map(ord, completion)) % 11))
                .with_checkpointing(directory / "checkpoint", every_updates=1))
        plan.start_now()
        state = {"policy": item.model.module.state_dict(), "reference": reference.state_dict()}
        if algorithm == "ppo":
            state["critic"] = objective.critic.module.state_dict()
        torch.save(state, directory / ("rank{}.pt".format(rank)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
def test_two_process_ddp_matches_global_token_mean(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(51)
    item = CausalLanguageModel(tiny(), ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.01)
    TrainingPlan().add_training_item(item).with_data_loader(data(batch_size=4)).start_now()
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, value in item.model.state_dict().items():
        torch.testing.assert_close(rank0[key], rank1[key], rtol=0, atol=0)
        torch.testing.assert_close(value, rank0[key], rtol=1e-5, atol=1e-7)
    state = torch.load(tmp_path / "checkpoint" / "latest.pt", weights_only=True)
    assert state["world_size"] == 2 and len(state["ranks"]) == 2


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("algorithm", ["grpo", "ppo"])
def test_distributed_rl_keeps_policy_reference_and_critic_synchronized(tmp_path, algorithm):
    torch.multiprocessing.spawn(_ddp_rl_worker, args=(str(tmp_path), algorithm), nprocs=2, join=True)
    left = torch.load(tmp_path / "rank0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for group in left:
        for key in left[group]:
            torch.testing.assert_close(left[group][key], right[group][key], atol=0, rtol=0)


class InvalidObjective(CausalLMObjective):
    def __call__(self, model, batch):
        result = super().__call__(model, batch)
        result.loss_sum = result.loss_sum * float("nan")
        return result


def test_invalid_loss_propagates_and_does_not_overwrite_checkpoint(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"known-good-placeholder")
    item = CausalLanguageModel(tiny()).with_objective(InvalidObjective()).with_optimizer(torch.optim.SGD, lr=0.01)
    before = copy.deepcopy(item.model.state_dict())
    plan = TrainingPlan().add_training_item(item).with_data_loader(data()).with_checkpointing(tmp_path, every_updates=1)
    with pytest.raises(FloatingPointError):
        plan.start_now()
    assert checkpoint.read_bytes() == b"known-good-placeholder"
    assert item.optimizer_updates == 0
    for key, value in item.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_resume_after_short_tail_does_not_repeat_epoch(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(33)
    initial = tiny()
    def build(model):
        item = (CausalLanguageModel(model, ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.01)
                .with_accumulate_grads(3))
        return TrainingPlan().add_training_item(item).with_data_loader(data()).repeat_epochs(2), item
    baseline, expected = build(copy.deepcopy(initial))
    baseline.start_now()
    partial, _ = build(copy.deepcopy(initial))
    partial.with_checkpointing(tmp_path, every_updates=1).limit_updates(2).start_now()
    resumed, actual = build(tiny())
    resumed.resume_from(partial.last_checkpoint).start_now()
    assert actual.optimizer_updates == expected.optimizer_updates == 4
    for left, right in zip(expected.model.parameters(), actual.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_changed_collator_config_rejects_resume(tmp_path):
    torch.set_num_threads(1)
    item = CausalLanguageModel(tiny(), ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.01)
    plan = TrainingPlan().add_training_item(item).with_data_loader(data()).with_checkpointing(tmp_path).limit_updates(1)
    plan.start_now()
    new = CausalLanguageModel(tiny(), ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.01)
    loader = data()
    loader.collate.max_length = 16
    with pytest.raises(ValueError, match="data configuration"):
        TrainingPlan().add_training_item(new).with_data_loader(loader).resume_from(plan.last_checkpoint).start_now()


def test_callback_update_counts_match_scheduler():
    torch.set_num_threads(1)
    events = []
    class Recorder:
        is_shared = False
        def on_optimization_step_end(self, context):
            events.append(context["optimizer_updates"])
    item = (CausalLanguageModel(tiny()).with_optimizer(torch.optim.SGD, lr=0.01)
            .with_accumulate_grads(3).with_callbacks(Recorder()))
    item.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5)
    TrainingPlan().add_training_item(item).with_data_loader(data()).start_now()
    assert events == [1, 2]
    assert item.optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)


def test_with_batch_size_applies_to_native_iterator():
    torch.set_num_threads(1)
    item = CausalLanguageModel(tiny()).with_optimizer(torch.optim.SGD, lr=0.01)
    loader = data()
    TrainingPlan().add_training_item(item).with_data_loader(loader).with_batch_size(4).start_now()
    assert item.optimizer_updates == 1


class _OverflowGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class OverflowOnce(CausalLMObjective):
    def __init__(self):
        self.calls = 0

    def __call__(self, model, batch):
        result = super().__call__(model, batch)
        self.calls += 1
        if self.calls == 1:
            result.loss_sum = _OverflowGradient.apply(result.loss_sum)
        return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_overflow_skips_scheduler_and_recovers():
    torch.set_num_threads(1)
    item = (CausalLanguageModel(tiny().cuda()).with_objective(OverflowOnce())
            .with_optimizer(torch.optim.SGD, lr=0.01).with_precision("fp16"))
    item.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5)
    TrainingPlan().add_training_item(item).with_data_loader(data(batch_size=2)).start_now()
    assert item.history[0]["updated"] is False
    assert item.history[1]["updated"] is True
    assert item.optimizer_updates == 1
    assert item.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

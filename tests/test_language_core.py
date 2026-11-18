import copy
import math
import os

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import numpy as np
import pytest
import torch

from trident.data.pipeline import (ByteTokenizer, CausalLMCollator, Dataset, Iterator,
                                   PackedCollator, PreferenceCollator, SFTCollator)
from trident.optims.language import (CausalLM, CausalLMConfig, CausalLanguageModel,
                                    DPO, Distillation, SFT, TrainingPipeline)
from trident.optims.language.lora import merge_lora
from trident.optims.language.runner import move_batch
from trident.optims.trainers import TrainingPlan


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def tiny(**kwargs):
    values = dict(vocab_size=272, hidden_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, intermediate_size=32,
                  max_position_embeddings=128)
    values.update(kwargs)
    return CausalLM(CausalLMConfig(**values))


def batches(batch_size=1):
    return Iterator(Dataset([{"text": text} for text in ("abc", "abcdef", "a", "xy", "foo")]),
                    batch_size=batch_size, collate=CausalLMCollator(ByteTokenizer(), max_length=32))


def plan_for(network, loader, accumulation=1):
    item = CausalLanguageModel(network, ByteTokenizer()).with_optimizer(torch.optim.SGD, lr=0.05).with_accumulate_grads(accumulation)
    return TrainingPlan().add_training_item(item).with_data_loader(loader), item


def test_attention_causality_and_kv_cache():
    model = tiny().eval()
    ids = torch.tensor([[1, 30, 31, 32, 33]])
    full = model(ids)["logits"]
    changed = ids.clone()
    changed[0, -1] = 100
    torch.testing.assert_close(full[:, :-1], model(changed)["logits"][:, :-1])
    first = model(ids[:, :3], use_cache=True)
    tail = model(ids[:, 3:], past_key_values=first["past_key_values"], use_cache=True)
    torch.testing.assert_close(full[:, 3:], tail["logits"], atol=1e-6, rtol=1e-5)


def test_padding_eos_and_assistant_spans():
    tokenizer = ByteTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    collator = CausalLMCollator(tokenizer)
    data = collator([{"text": "a"}, {"text": "abc"}])
    assert data["labels"][0, 2] == tokenizer.eos_token_id
    assert data["labels"][0, 3] == -100
    sample = {"messages": [{"role": "user", "content": "literal assistant <|im_start|>assistant\n"},
                            {"role": "assistant", "content": "OK"},
                            {"role": "tool", "content": "OBSERVATION"}]}
    output = SFTCollator(tokenizer)([sample])
    supervised = output["labels"][0]
    decoded = tokenizer.decode(supervised[supervised != -100])
    assert decoded == "OK<|im_end|>"


def test_packing_isolates_documents():
    tokenizer = ByteTokenizer()
    collator = PackedCollator(CausalLMCollator(tokenizer, max_length=16), max_length=32)
    batch = move_batch(collator([{"text": "abc"}, {"text": "xyz"}]), "cpu")
    model = tiny().eval()
    output = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"], segment_ids=batch["segment_ids"])["logits"]
    batch["input_ids"][0, 1] = 99
    changed = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"], segment_ids=batch["segment_ids"])["logits"]
    torch.testing.assert_close(output[:, 5:], changed[:, 5:])
    assert batch["labels"][0, 5] == -100


def test_external_kwargs_model_cannot_silently_ignore_packing():
    from trident.optims.language.objectives import forward_model
    class ExternalModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return tiny()(input_ids)
    batch = move_batch(PackedCollator(CausalLMCollator(ByteTokenizer(), max_length=16), 32)(
        [{"text": "a"}, {"text": "b"}]), "cpu")
    with pytest.raises(ValueError, match="packing"):
        forward_model(ExternalModel(), batch)


@pytest.mark.parametrize("experts", [0, 3])
def test_accumulation_equals_large_batch_including_short_tail(experts):
    large = tiny(num_experts=experts)
    micro = copy.deepcopy(large)
    full_plan, full = plan_for(large, batches(batch_size=2))
    micro_plan, small = plan_for(micro, batches(), accumulation=2)
    full_plan.start_now()
    micro_plan.start_now()
    assert full.optimizer_updates == small.optimizer_updates == 3
    assert full.valid_tokens == small.valid_tokens
    for left, right in zip(large.parameters(), micro.parameters()):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)


def test_lora_only_updates_adapters_and_merge_is_equivalent():
    model = tiny()
    base = copy.deepcopy(model.state_dict())
    item = CausalLanguageModel(model).with_lora(rank=2, alpha=4).with_optimizer(torch.optim.AdamW, lr=0.01)
    TrainingPlan().add_training_item(item).with_data_loader(batches()).start_now()
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, base[name.replace(".base.", ".")], rtol=0, atol=0)
    model.eval()
    ids = torch.tensor([[1, 30, 31]])
    before = model(ids)["logits"]
    merged = merge_lora(copy.deepcopy(model))
    torch.testing.assert_close(before, merged(ids)["logits"], atol=1e-6, rtol=1e-5)


def test_dpo_and_distillation_keep_auxiliary_models_frozen():
    model = tiny()
    reference = copy.deepcopy(model)
    objective = DPO(reference)
    objective.prepare(model, "cpu")
    pairs = PreferenceCollator(ByteTokenizer())([{"prompt": "Q", "chosen": "yes", "rejected": "no"}])
    result = objective(model, move_batch(pairs, "cpu"))
    assert float(result.loss.detach()) == pytest.approx(math.log(2), rel=1e-5)
    result.loss.backward()
    assert all(value.grad is None and not value.requires_grad for value in reference.parameters())
    teacher = copy.deepcopy(model)
    teacher.zero_grad(set_to_none=True)
    distillation = Distillation(teacher, alpha=1)
    distillation.prepare(model, "cpu")
    distilled = distillation(model, move_batch(next(iter(batches())), "cpu"))
    assert abs(float(distilled.loss.detach())) < 1e-6
    with pytest.raises(ValueError, match="share"):
        DPO(model).prepare(model, "cpu")


def test_preference_mismatched_prompt_and_truncation_rejected():
    collator = PreferenceCollator(ByteTokenizer(), max_length=8)
    with pytest.raises(ValueError, match="truncated"):
        collator([{"prompt": "long prompt", "chosen": "a", "rejected": "b"}])
    with pytest.raises(ValueError, match="identical"):
        collator([{"chosen": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "a"}],
                   "rejected": [{"role": "user", "content": "y"}, {"role": "assistant", "content": "b"}]}])


def test_exact_resume_with_rng_and_scheduler(tmp_path):
    initial = tiny(attention_dropout=0.15)
    baseline_model, partial_model = copy.deepcopy(initial), copy.deepcopy(initial)
    baseline_plan, baseline = plan_for(baseline_model, batches(), accumulation=2)
    baseline.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9)
    baseline_plan.repeat_epochs(2)
    torch.manual_seed(99)
    baseline_plan.start_now()
    partial_plan, partial = plan_for(partial_model, batches(), accumulation=2)
    partial.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9)
    partial_plan.repeat_epochs(2).with_checkpointing(tmp_path / "ckpt", every_updates=1).limit_updates(2)
    torch.manual_seed(99)
    partial_plan.start_now()
    resume_plan, resumed = plan_for(tiny(attention_dropout=0.15), batches(), accumulation=2)
    resumed.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9)
    resume_plan.repeat_epochs(2).resume_from(partial_plan.last_checkpoint).start_now()
    assert baseline.optimizer_updates == resumed.optimizer_updates == 6
    assert baseline.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"]
    for left, right in zip(baseline_model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_empty_targets_skip_update_and_scheduler():
    model = tiny()
    plan, item = plan_for(model, [{"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, -100]])}])
    item.with_learning_rate_scheduler(torch.optim.lr_scheduler.StepLR, step_size=1)
    plan.start_now()
    assert item.optimizer_updates == 0
    assert not item.history[-1]["updated"]
    assert item.optimizer.param_groups[0]["lr"] == 0.05


def test_moe_yarn_checkpointing_backward():
    model = tiny(num_experts=3, num_experts_per_tok=2,
                 rope_scaling={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 64})
    model.gradient_checkpointing_enable()
    plan, item = plan_for(model, batches())
    plan.start_now()
    assert all(math.isfinite(row["loss"]) for row in item.history)


def test_stage_factories_receive_completed_artifacts(tmp_path):
    calls = []
    def first(previous):
        assert previous is None
        calls.append("pretrain")
        return plan_for(tiny(), batches())[0]
    def second(previous):
        assert previous.checkpoint.exists()
        calls.append("sft")
        model = CausalLM.from_pretrained(previous.model_directory)
        return plan_for(model, batches())[0]
    pipeline = TrainingPipeline(tmp_path / "pipeline").add_stage("pretrain", first).add_stage("sft", second)
    assert len(pipeline.start_now()) == 2
    assert calls == ["pretrain", "sft"]
    pipeline.start_now(resume=True)
    assert calls == ["pretrain", "sft"]



def test_masked_log_probs_and_rollout_behavior_distribution():
    from trident.optims.language.objectives import token_log_probs
    from trident.optims.language.rollout import TorchRolloutEngine

    logits = torch.tensor([[[-torch.inf, -torch.inf], [0., 1.]]], requires_grad=True)
    result, valid = token_log_probs(logits, torch.tensor([[1, -100]]))
    assert not valid.any() and result.item() == 0
    (gradient,) = torch.autograd.grad(result.sum(), logits)
    assert torch.equal(gradient, torch.zeros_like(gradient))

    engine = TorchRolloutEngine(ByteTokenizer())
    engine.update_policy(tiny(), 0)
    rollout = engine.rollout(["Q"], num_generations=1, max_new_tokens=1, top_k=1)
    labels = rollout.tensors["labels"][:, 1:]
    assert rollout.tensors["old_log_probs"][labels != -100].tolist() == [0.0]

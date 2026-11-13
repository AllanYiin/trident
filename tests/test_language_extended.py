import copy
import json
import os
import time

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import pytest
import torch

from trident.data.pipeline import (ByteTokenizer, CausalLMCollator, Dataset, Iterator,
                                   jsonl_dataset, load_tokenizer, train_bpe_tokenizer)
from trident.optims.language import (AgenticRolloutEngine, CausalLM, CausalLMConfig,
                                    CausalLanguageModel, GRPO, PPO, RLTrainingPlan,
                                    ToolRegistry, TorchRolloutEngine, ValueModel, parse_response)
from trident.optims.language.checkpoint import atomic_torch_save, read_checkpoint
from trident.optims.language.interop import to_huggingface, to_hf_tokenizer
from trident.optims.trainers import TrainingPlan


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def tiny(**kwargs):
    values = dict(vocab_size=272, hidden_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, intermediate_size=32,
                  max_position_embeddings=512)
    values.update(kwargs)
    return CausalLM(CausalLMConfig(**values))


def tool_add(a, b):
    return a + b


def tool_slow():
    time.sleep(10)


def test_bpe_and_jsonl_roundtrip(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('\n'.join(json.dumps({"text": text}, ensure_ascii=False) for text in ["你好世界", "hello world"]), encoding="utf-8")
    data = jsonl_dataset(path)
    tokenizer = train_bpe_tokenizer((row["text"] for row in data), vocab_size=300, min_frequency=1)
    text = "你好 hello <think>OK</think>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    loaded = load_tokenizer(tmp_path / "tokenizer")
    assert loaded.encode(text) == tokenizer.encode(text)
    assert len(list(data)) == 2


@pytest.mark.parametrize("experts", [0, 2])
def test_huggingface_logits_and_export(experts, tmp_path):
    pytest.importorskip("transformers")
    native = tiny(num_experts=experts).eval()
    hf = to_huggingface(native)
    ids = torch.tensor([[1, 40, 41, 2]])
    with torch.no_grad():
        expected = native(ids)["logits"]
        actual = hf(ids).logits
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
    item = CausalLanguageModel(native, ByteTokenizer())
    path = item.export(tmp_path / "hf", format="huggingface")
    loaded = CausalLanguageModel.from_huggingface(path)
    loaded.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded.model(ids).logits, atol=1e-6, rtol=1e-5)
    tokenizer = to_hf_tokenizer(ByteTokenizer())
    text = "繁中 test <think>ok</think>"
    assert tokenizer.encode(text, add_special_tokens=False) == ByteTokenizer().encode(text)


@pytest.mark.parametrize("algorithm", ["grpo", "cispo", "ppo"])
def test_online_rl_updates_and_saves(algorithm, tmp_path):
    network = tiny()
    reference = copy.deepcopy(network)
    if algorithm == "ppo":
        critic = ValueModel(copy.deepcopy(network))
        before_critic = copy.deepcopy(critic.state_dict())
        objective = PPO(reference, critic)
    else:
        objective = GRPO(reference, loss_type=algorithm)
    item = CausalLanguageModel(network, ByteTokenizer()).with_objective(objective).with_optimizer(torch.optim.AdamW, lr=0.001)
    before = copy.deepcopy(network.state_dict())
    loader = Iterator(Dataset([{"prompt": "Q"}, {"prompt": "R"}]), batch_size=1, collate=list)
    plan = (RLTrainingPlan().add_training_item(item).with_data_loader(loader)
            .with_rollout(num_generations=2, max_new_tokens=4)
            .with_reward(lambda prompt, completion, sample: float(sum(map(ord, completion)) % 11))
            .with_policy_updates(batch_size=2).with_checkpointing(tmp_path / algorithm, every_updates=1))
    plan.start_now()
    assert item.optimizer_updates == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in item.history)
    assert all(value.grad is None and not value.requires_grad for value in reference.parameters())
    state = read_checkpoint(plan.last_checkpoint)
    assert state["optimizer_updates"] == 2
    assert any(not torch.equal(value, before[key]) for key, value in network.state_dict().items())
    if algorithm == "ppo":
        assert any(not torch.equal(value, before_critic[key]) for key, value in critic.state_dict().items())


def test_grpo_equal_rewards_and_old_policy_are_finite():
    model = tiny()
    engine = TorchRolloutEngine(ByteTokenizer())
    engine.update_policy(model, 3)
    rollout = engine.rollout([{"prompt": "Q"}], num_generations=2, max_new_tokens=3)
    objective = GRPO(copy.deepcopy(model))
    objective.prepare(model, "cpu")
    metrics = objective.prepare_rollout(rollout, torch.ones(2))
    assert metrics["degenerate_groups"] == 1
    assert torch.equal(rollout.tensors["advantages"], torch.zeros(2))
    result = objective(model, rollout.tensors)
    assert torch.isfinite(result.loss)
    assert not rollout.tensors["old_log_probs"].requires_grad


@pytest.mark.parametrize("algorithm", ["grpo", "ppo"])
def test_rl_resume_reproduces_next_rollout_and_optimizer(algorithm, tmp_path):
    initial = tiny()
    initial_critic = ValueModel(copy.deepcopy(initial))
    def build():
        network, reference = copy.deepcopy(initial), copy.deepcopy(initial)
        objective = PPO(reference, copy.deepcopy(initial_critic)) if algorithm == "ppo" else GRPO(reference)
        item = CausalLanguageModel(network, ByteTokenizer()).with_objective(objective).with_optimizer(torch.optim.AdamW, lr=0.001)
        loader = Iterator(Dataset([{"prompt": "Q"}, {"prompt": "R"}]), batch_size=1, collate=list)
        plan = (RLTrainingPlan().add_training_item(item).with_data_loader(loader)
                .with_rollout(num_generations=2, max_new_tokens=3)
                .with_reward(lambda prompt, completion, sample: float(sum(map(ord, completion)) % 11))
                .with_policy_updates(batch_size=2))
        return plan, item
    baseline, expected = build()
    torch.manual_seed(333)
    baseline.start_now()
    partial, _ = build()
    torch.manual_seed(333)
    partial.with_checkpointing(tmp_path, every_updates=1).limit_updates(1).start_now()
    resumed, actual = build()
    resumed.resume_from(partial.last_checkpoint).start_now()
    assert actual.optimizer_updates == expected.optimizer_updates == 2
    for left, right in zip(expected.model.parameters(), actual.model.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    if algorithm == "ppo":
        for left, right in zip(expected.objective.critic.parameters(), actual.objective.critic.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_adapter_only_save_load_and_training_model_survives_export(tmp_path):
    initial = tiny()
    item = (CausalLanguageModel(copy.deepcopy(initial), ByteTokenizer())
            .with_lora(rank=2, alpha=4).with_optimizer(torch.optim.AdamW, lr=0.01))
    loader = Iterator(Dataset([{"text": "a"}, {"text": "b"}]), collate=CausalLMCollator(ByteTokenizer()))
    TrainingPlan().add_training_item(item).with_data_loader(loader).start_now()
    item.save_adapter(tmp_path / "adapter.pt")
    loaded = CausalLanguageModel(copy.deepcopy(initial), ByteTokenizer()).load_adapter(tmp_path / "adapter.pt")
    ids = torch.tensor([[1, 30, 31]])
    item.model.eval()
    loaded.model.eval()
    torch.testing.assert_close(item.model(ids)["logits"], loaded.model(ids)["logits"], atol=0, rtol=0)
    before = copy.deepcopy(item.model.state_dict())
    item.export(tmp_path / "merged")
    for key, value in item.model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_incomplete_tool_and_thinking_response():
    assert parse_response("<think>unfinished")["reasoning_content"] == "unfinished"
    with pytest.raises(ValueError, match="incomplete"):
        parse_response('<tool_call>{"name":"x"}')


def test_tool_execution_timeout_and_registry():
    registry = ToolRegistry().register("add", tool_add, timeout=10).register("slow", tool_slow, timeout=0.05)
    assert registry.execute({"function": {"name": "add", "arguments": '{"a":2,"b":3}'}}) == {"ok": True, "result": 5}
    assert not registry.execute({"function": {"name": "missing", "arguments": {}}})["ok"]
    assert registry.execute({"function": {"name": "slow", "arguments": {}}})["error"] == "tool timeout"


def test_agentic_observations_never_receive_labels(monkeypatch):
    import trident.optims.language.agent as agent_module
    tokenizer = ByteTokenizer()
    registry = ToolRegistry().register("add", tool_add)
    monkeypatch.setattr(registry, "execute", lambda call: {"ok": True, "result": "OBSERVATION"})
    turns = iter(['<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call><|im_end|>', '5<|im_end|>'])
    def fake_generate(policy, ids, **kwargs):
        return torch.cat((ids, torch.tensor([tokenizer.encode(next(turns))])), dim=-1)
    monkeypatch.setattr(agent_module, "generate_tokens", fake_generate)
    engine = AgenticRolloutEngine(tokenizer, registry)
    engine.update_policy(tiny(max_position_embeddings=1024), 0)
    rollout = engine.rollout([{"messages": [{"role": "user", "content": "2+3"}]}], num_generations=1, max_new_tokens=100)
    supervised = rollout.tensors["labels"][0]
    text = tokenizer.decode(supervised[supervised != -100].tolist())
    assert "OBSERVATION" not in text
    assert "tool_call" in text and "5" in text
    assert rollout.trajectories[0][-1]["stop"] == "answer"


def test_chat_api_and_sse(monkeypatch):
    from fastapi.testclient import TestClient
    from trident.optims.language.serving import create_chat_app
    tokenizer = ByteTokenizer()
    item = CausalLanguageModel(tiny(), tokenizer)
    def fake_generate(ids, **kwargs):
        return torch.cat((ids, torch.tensor([tokenizer.encode('<think>calc</think>5<|im_end|>')])), dim=-1)
    monkeypatch.setattr(item, "generate", fake_generate)
    client = TestClient(create_chat_app(item))
    request = {"model": "trident", "messages": [{"role": "user", "content": "2+3"}], "max_tokens": 32}
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "5" and message["reasoning_content"] == "calc"
    stream = client.post("/v1/chat/completions", json=dict(request, stream=True))
    assert "data: [DONE]" in stream.text
    assert client.post("/v1/chat/completions", json=dict(request, model="wrong")).status_code == 404


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cuda_mixed_precision_training(precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unavailable")
    item = (CausalLanguageModel(tiny().cuda(), ByteTokenizer()).with_precision(precision)
            .with_optimizer(torch.optim.AdamW, lr=0.001).with_accumulate_grads(2).with_grad_clipping(1))
    loader = Iterator(Dataset([{"text": "hello"}, {"text": "world"}, {"text": "short tail"}]),
                      collate=CausalLMCollator(ByteTokenizer()))
    TrainingPlan().add_training_item(item).with_data_loader(loader).start_now()
    assert item.optimizer_updates > 0
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in item.history)

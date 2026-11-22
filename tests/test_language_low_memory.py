import copy
import json

import pytest
import torch

from trident.optims.language import CausalLanguageModel, CausalLM, CausalLMConfig, SFT, load_pretrained_weights
from trident.optims.language import pretrained
from trident.optims.language.quantization import NF4Linear


def model(**kwargs):
    return CausalLM(CausalLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, intermediate_size=64, vocab_size=257, **kwargs))


def test_indexed_load_and_partial_reads(tmp_path, monkeypatch):
    net = model()
    net.save_pretrained(tmp_path, max_shard_size="1KiB")
    index = json.loads((tmp_path / "model.pt.index.json").read_text())["weight_map"]
    other = CausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert other.lm_head.weight is other.model.embed_tokens.weight
    assert other.lm_head.weight.dtype == torch.bfloat16
    assert all(not t.is_meta for t in other.buffers())
    ids = torch.tensor([[1, 9, 12]])
    torch.testing.assert_close(CausalLM.from_pretrained(tmp_path)(ids)["logits"], net(ids)["logits"])
    read, mapped = set(), pretrained._mapped
    def record(path):
        read.add(path.name)
        return mapped(path)
    monkeypatch.setattr(pretrained, "_mapped", record)
    target = torch.nn.Linear(32, 32, bias=False)
    prefix = "model.layers.1.self_attn.q_proj"
    report = load_pretrained_weights(target, tmp_path, source_prefix=prefix)
    assert report == dict(loaded=["weight"], mismatched=[], missing=[], unexpected=[])
    assert read == {index[prefix + ".weight"]}
    torch.testing.assert_close(target.weight, net.get_submodule(prefix).weight)
    wrong = torch.nn.Linear(32, 2, bias=False)
    before = wrong.weight.clone()
    assert load_pretrained_weights(wrong, tmp_path, prefix, strict=False)["mismatched"] == ["weight"]
    with pytest.raises(ValueError, match="incompatible"):
        load_pretrained_weights(wrong, tmp_path, prefix)
    torch.testing.assert_close(wrong.weight, before)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_chunked_checkpoint_gradients_and_generation(device):
    torch.manual_seed(4)
    net = model().to(device)
    small = copy.deepcopy(net)
    item = CausalLanguageModel(small).with_gradient_checkpointing(offload_to_cpu=True)
    ids = torch.tensor([[1, 12, 7, 10, 5], [2, 4, 6, 5, 9]], device=device)
    labels = ids.clone()
    labels[0, 1:4] = -100
    batch = {"input_ids": ids, "labels": labels}
    widths = []
    small.lm_head.register_forward_pre_hook(lambda mod, inputs: widths.append(inputs[0].shape[-2]))
    reference = SFT()(net, batch)
    chunked = SFT(logits_chunk_size=2)(item.model, batch)
    assert chunked.weight == reference.weight and "logits" not in chunked.outputs
    torch.testing.assert_close(chunked.loss, reference.loss)
    chunked.loss.backward()
    reference.loss.backward()
    for actual, expected in zip(small.parameters(), net.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-6)
    assert max(widths) == 2
    widths.clear()
    generated = small.generate(ids, max_new_tokens=2)
    assert widths and set(widths) == {1}
    torch.testing.assert_close(generated[:, ids.shape[1]], net(ids)["logits"][:, -1].argmax(-1))
    empty = SFT(logits_chunk_size=2)(small, {"input_ids": ids, "labels": labels.fill_(-100)})
    assert empty.loss_sum.item() == 0 and empty.weight == 0
    empty.loss_sum.backward()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_nf4_packed_reload_and_qlora(tmp_path, device, monkeypatch):
    net = model()
    net.save_pretrained(tmp_path)
    packed = CausalLM.from_pretrained(tmp_path, device, quantization_config={"format": "nf4"})
    ids = torch.tensor([[2, 5, 19]], device=device)
    with torch.no_grad():
        reference = net(ids.cpu())["logits"].to(device)
        output = packed(ids)["logits"]
    assert (output - reference).norm() < 0.15 * reference.norm()
    assert not any(p.requires_grad for p in packed.parameters())
    packed.save_pretrained(tmp_path / "packed", max_shard_size=4096)
    monkeypatch.setattr(NF4Linear, "pack", lambda *args: pytest.fail("requantized a packed checkpoint"))
    loaded = CausalLM.from_pretrained(tmp_path / "packed", device)
    torch.testing.assert_close(loaded(ids)["logits"], output, atol=0, rtol=0)
    item = (CausalLanguageModel(loaded).with_lora(rank=4, adapter_dtype=torch.float32)
            .with_gradient_checkpointing(offload_to_cpu=True)
            .with_objective(SFT(logits_chunk_size=1)).with_optimizer(lr=.01))
    item.prepare()
    item.compute({"input_ids": ids, "labels": ids}).loss.backward()
    assert all("lora_" in k for k, p in loaded.named_parameters() if p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().any() for p in loaded.parameters())
    item.optimizer.step()
    item.export(tmp_path / "merged")
    merged = CausalLM.from_pretrained(tmp_path / "merged", device)
    torch.testing.assert_close(loaded(ids)["logits"], merged(ids)["logits"], atol=2e-6, rtol=1e-4)


def test_accelerate_cpu_auto(tmp_path):
    pytest.importorskip("accelerate")
    net = model()
    net.save_pretrained(tmp_path, max_shard_size="32KiB")
    loaded = CausalLM.from_pretrained(tmp_path, device_map="auto", max_memory={"cpu": "1GiB"})
    assert set(loaded.hf_device_map.values()) == {"cpu"}
    ids = torch.tensor([[1, 3, 4]])
    torch.testing.assert_close(net(ids)["logits"], loaded(ids)["logits"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed CUDA/CPU/disk placement")
@pytest.mark.parametrize("packed", [False, True])
def test_accelerate_disk_generation(tmp_path, packed):
    pytest.importorskip("accelerate")
    net = model()
    net.save_pretrained(tmp_path, max_shard_size="32KiB")
    if packed:
        net = CausalLM.from_pretrained(tmp_path, quantization_config={"format": "nf4"})
        net.save_pretrained(tmp_path, max_shard_size="32KiB")
    placement = {"model.embed_tokens": 0, "model.layers.0": "cpu", "model.layers.1": "disk",
                 "model.norm": 0, "lm_head": 0}
    dtype = torch.float16 if packed else torch.float32
    loaded = CausalLM.from_pretrained(tmp_path, device_map=placement, offload_folder=tmp_path / "disk",
                                      dtype=dtype)
    item = CausalLanguageModel(loaded)
    ids = torch.tensor([[1, 12, 4]])
    torch.testing.assert_close(item.generate(ids.cuda(), max_new_tokens=3).cpu(),
                               net.to("cuda", dtype=dtype).generate(ids.cuda(), max_new_tokens=3).cpu())
    with pytest.raises(ValueError, match="cannot be trained"):
        item.with_optimizer()
    with pytest.raises(RuntimeError, match="device_map"):
        item.to("cpu")

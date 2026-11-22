"""Measure allocation peaks of the two vocabulary projection paths on the same GPU."""
import pytest
import torch

from trident.optims.language import CausalLM, CausalLMConfig, SFT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator required")
def test_vocabulary_projection_peak():
    net = CausalLM(CausalLMConfig(vocab_size=32768, hidden_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, intermediate_size=256)).cuda()
    ids = torch.arange(512, device="cuda").unsqueeze(0)
    batch = {"input_ids": ids, "labels": ids}

    def peak(run):
        net.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start

    full = peak(lambda: SFT()(net, batch).loss.backward())
    chunked = peak(lambda: SFT(logits_chunk_size=32)(net, batch).loss.backward())
    with torch.no_grad():
        all_logits = peak(lambda: net(ids, use_cache=True))
        last = peak(lambda: net.generate(ids, max_new_tokens=1))
    print(f"CUDA peak bytes: full_loss={full} chunked_loss={chunked} full_prompt={all_logits} last_token={last}")
    assert chunked < full
    assert last < all_logits

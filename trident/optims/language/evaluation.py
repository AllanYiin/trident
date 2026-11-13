"""Small local evaluation adapters; benchmark datasets are supplied by callers."""
import math

import torch

from .checkpoint import capture_rng, restore_rng
from .runner import evaluate


def perplexity(item, loader):
    from .objectives import CausalLMObjective, SFT
    if type(item.objective) not in (CausalLMObjective, SFT):
        raise ValueError("perplexity requires a token cross-entropy objective")
    metrics = evaluate(item, loader)
    # Auxiliary losses must not be misreported as language-model perplexity.
    model = item.model.module if hasattr(item.model, "module") else item.model
    if getattr(getattr(model, "config", None), "num_experts", 0) or item.extra_losses:
        raise ValueError("use token NLL evaluation without auxiliary losses for perplexity")
    metrics["perplexity"] = math.exp(metrics["loss"]) if metrics["loss"] < 700 else math.inf
    return metrics


@torch.no_grad()
def evaluate_generation(item, samples, score_fn, max_new_tokens=32):
    """score_fn(completion, sample) can implement exact match or benchmark scoring."""
    from .rollout import prompt_text
    from trident.data.pipeline.language import ChatTemplate
    rows, scores = [], []
    state = capture_rng()
    try:
        for sample in samples:
            prompt = prompt_text(sample, ChatTemplate())
            ids = torch.tensor([item.tokenizer.encode(prompt, add_special_tokens=False)], device=item.device)
            result = item.generate(ids, max_new_tokens=max_new_tokens)
            completion = item.tokenizer.decode(result[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            score = float(score_fn(completion, sample))
            rows.append({"prompt": prompt, "completion": completion, "score": score})
            scores.append(score)
    finally:
        restore_rng(state)
    if not scores:
        raise ValueError("evaluation requires at least one sample")
    return {"score": sum(scores) / len(scores), "samples": rows}

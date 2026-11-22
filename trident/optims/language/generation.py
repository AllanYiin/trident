"""Shared native/Hugging Face generation with explicit padding and sampling."""
import inspect
import torch

from .objectives import forward_model, unwrap_model


def sample_token(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None, *, return_log_probs=False):
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    logits = logits.float()
    if temperature == 0:
        tokens = logits.argmax(-1)
        return (tokens, torch.zeros_like(tokens, dtype=logits.dtype)) if return_log_probs else tokens
    logits = logits / temperature
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1:
        sorted_logits, order = logits.sort(-1, descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.scatter(1, order, sorted_logits.masked_fill(remove, float("-inf")))
    log_probs = logits.log_softmax(-1)
    tokens = torch.multinomial(log_probs.exp(), 1, generator=generator)
    if return_log_probs:
        return tokens.squeeze(-1), log_probs.gather(-1, tokens).squeeze(-1)
    return tokens.squeeze(-1)


@torch.no_grad()
def generate_tokens(model, input_ids, attention_mask=None, max_new_tokens=32,
                    temperature=0.0, top_k=0, top_p=1.0, eos_token_id=None,
                    pad_token_id=None, generator=None, *, return_log_probs=False):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    config = getattr(unwrap_model(model), "config", None)
    capacity = getattr(config, "max_position_embeddings", None)
    if capacity is not None and input_ids.shape[-1] + max_new_tokens > capacity:
        raise ValueError("prompt plus max_new_tokens exceeds model context capacity")
    eos_token_id = eos_token_id if eos_token_id is not None else getattr(config, "eos_token_id", None)
    pad_token_id = pad_token_id if pad_token_id is not None else getattr(config, "pad_token_id", 0)
    pad_token_id = 0 if pad_token_id is None else pad_token_id
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.long()
    if input_ids.ndim != 2 or mask.shape != input_ids.shape or not bool(mask.any(-1).all()):
        raise ValueError("generation requires a nonempty prompt per sample")
    # Canonical left padding gives every sample a last real prompt token.
    output = torch.full_like(input_ids, pad_token_id)
    left_mask = torch.zeros_like(mask)
    for index, row in enumerate(input_ids):
        tokens = row[mask[index].bool()]
        output[index, -len(tokens):] = tokens
        left_mask[index, -len(tokens):] = 1
    mask = left_mask
    active = torch.ones(output.shape[0], dtype=torch.bool, device=output.device)
    eos_ids = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
    was_training = model.training
    model.eval()
    cache = None
    sampled_log_probs = []
    last_only = "logits_to_keep" in inspect.signature(unwrap_model(model).forward).parameters
    try:
        for _ in range(max_new_tokens):
            data = {"input_ids": output if cache is None else output[:, -1:], "attention_mask": mask}
            extra = {"use_cache": True}
            if last_only:
                extra["logits_to_keep"] = 1
            if cache is not None:
                extra["past_key_values"] = cache
            result = forward_model(model, data, **extra)
            sample = sample_token(result["logits"][:, -1], temperature, top_k, top_p,
                                  generator, return_log_probs=return_log_probs)
            if return_log_probs:
                token, log_prob = sample
                sampled_log_probs.append(log_prob.masked_fill(~active, 0))
            else:
                token = sample
            token = torch.where(active, token, torch.full_like(token, pad_token_id))
            output = torch.cat((output, token[:, None]), dim=-1)
            mask = torch.cat((mask, active.long()[:, None]), dim=-1)
            for eos in eos_ids:
                active = active & (token != eos)
            cache = result.get("past_key_values")
            if not bool(active.any()):
                break
    finally:
        model.train(was_training)
    if return_log_probs:
        log_probs = torch.stack(sampled_log_probs, dim=1) if sampled_log_probs else output.new_empty(
            (output.shape[0], 0), dtype=torch.float32)
        return output, log_probs
    return output

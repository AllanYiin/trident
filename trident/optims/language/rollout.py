"""Versioned rollout records shared by local, external and tool-use engines."""
import copy
from dataclasses import dataclass

import torch

from trident.data.pipeline.language import ChatTemplate
from .generation import generate_tokens
from .objectives import forward_model, token_log_probs, unwrap_model


@dataclass
class RolloutBatch:
    tensors: dict
    prompts: list
    completions: list
    group_ids: torch.Tensor
    policy_version: int
    trajectories: list = None

    def validate(self):
        inputs = self.tensors["input_ids"]
        labels = self.tensors["labels"]
        logps = self.tensors["old_log_probs"]
        if inputs.ndim != 2 or labels.shape != inputs.shape or logps.shape != inputs[:, 1:].shape:
            raise ValueError("rollout input/label/log-probability shapes disagree")
        if len(inputs) != len(self.prompts) or len(inputs) != len(self.completions) or self.group_ids.numel() != len(inputs):
            raise ValueError("rollout metadata length mismatch")
        if not bool(torch.isfinite(logps).all()) or logps.requires_grad:
            raise ValueError("old_log_probs must be finite detached sampling-policy log probabilities")
        if not bool((labels[:, 1:] != -100).any(-1).all()):
            raise ValueError("each rollout needs at least one action token")
        return self


def prompt_text(sample, template):
    if isinstance(sample, str):
        return sample
    if "prompt" in sample and isinstance(sample["prompt"], str):
        return sample["prompt"]
    messages = sample.get("messages", sample.get("conversations", sample.get("prompt")))
    return template.render(messages, tools=sample.get("tools"), add_generation_prompt=True,
                           thinking=sample.get("thinking", False))[0]


class TorchRolloutEngine:
    def __init__(self, tokenizer, template=None):
        self.tokenizer = tokenizer
        self.template = template or ChatTemplate()
        self.policy = None
        self.policy_version = -1

    def update_policy(self, policy, version):
        if self.policy is None:
            self.policy = copy.deepcopy(unwrap_model(policy)).eval().requires_grad_(False)
        else:
            self.policy.load_state_dict(unwrap_model(policy).state_dict())
        self.policy_version = int(version)

    @torch.no_grad()
    def rollout(self, samples, num_generations=4, max_new_tokens=32,
                temperature=0.8, top_k=0, top_p=1.0):
        if self.policy is None:
            raise RuntimeError("synchronize the rollout policy before sampling")
        if num_generations < 1 or max_new_tokens < 1:
            raise ValueError("num_generations and max_new_tokens must be positive")
        device = next(self.policy.parameters()).device
        prompts = [prompt_text(sample, self.template) for sample in samples]
        encoded = [self.tokenizer.encode(text, add_special_tokens=False) for text in prompts]
        if not encoded or any(not row for row in encoded):
            raise ValueError("rollout prompts cannot be empty")
        width = max(map(len, encoded))
        ids = torch.full((len(samples) * num_generations, width), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            for generation in range(num_generations):
                ids[index * num_generations + generation, -len(row):] = torch.tensor(row, device=device)
                mask[index * num_generations + generation, -len(row):] = 1
        output = generate_tokens(self.policy, ids, mask, max_new_tokens, temperature, top_k, top_p,
                                 self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        completion = output[:, width:]
        completion_mask = torch.ones_like(completion)
        completions = []
        for index, row in enumerate(completion):
            ends = (row == self.tokenizer.eos_token_id).nonzero().flatten()
            length = int(ends[0]) + 1 if len(ends) else len(row)
            completion_mask[index, length:] = 0
            text_ids = row[:length - 1] if len(ends) else row[:length]
            completions.append(self.tokenizer.decode(text_ids.tolist(), skip_special_tokens=False))
        attention_mask = torch.cat((mask, completion_mask), dim=1)
        labels = torch.full_like(output, -100)
        labels[:, width:] = completion.masked_fill(~completion_mask.bool(), -100)
        tensors = {"input_ids": output, "attention_mask": attention_mask, "labels": labels}
        logits = forward_model(self.policy, tensors, use_cache=False)["logits"]
        tensors["old_log_probs"] = token_log_probs(logits, labels)[0].detach()
        return RolloutBatch(tensors, [prompt for prompt in prompts for _ in range(num_generations)],
                            completions, torch.arange(len(samples), device=device).repeat_interleave(num_generations),
                            self.policy_version).validate()


class ExternalRolloutEngine:
    """Explicit adapter for an inference service that supplies token log-probs.

    `sample_fn(samples, policy_version=..., **sampling)` returns RolloutBatch.
    `sync_fn(policy, version)` must complete weight synchronization before
    returning. A generic text-only chat-completions response is insufficient.
    """
    def __init__(self, sample_fn, sync_fn):
        self.sample_fn, self.sync_fn = sample_fn, sync_fn
        self.policy_version = -1

    def update_policy(self, policy, version):
        self.sync_fn(unwrap_model(policy), version)
        self.policy_version = int(version)

    def rollout(self, samples, **kwargs):
        output = self.sample_fn(samples, policy_version=self.policy_version, **kwargs)
        if not isinstance(output, RolloutBatch) or output.policy_version != self.policy_version:
            raise ValueError("external rollout has a stale policy version or invalid type")
        return output.validate()


class WeightedReward:
    """Compose scalar reward functions: function(prompt, completion, sample)."""
    def __init__(self, *weighted_functions):
        if not weighted_functions:
            raise ValueError("at least one reward function is required")
        self.functions = weighted_functions

    def __call__(self, prompt, completion, sample):
        return sum(weight * function(prompt, completion, sample) for weight, function in self.functions)

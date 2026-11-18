"""Native objectives; none of these classes owns an optimizer or calls backward."""
import inspect
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def forward_model(model, batch, **extra):
    keys = ("input_ids", "attention_mask", "position_ids", "segment_ids")
    arguments = {key: batch[key] for key in keys if key in batch}
    arguments.update(extra)
    signature = inspect.signature(unwrap_model(model).forward)
    if "segment_ids" in arguments and "segment_ids" not in signature.parameters:
        # Many HF models accept **kwargs but silently ignore segment IDs.
        # Such a forward cannot guarantee cross-document attention isolation.
        raise ValueError("this model does not support isolated sequence packing")
    if not any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values()):
        arguments = {key: value for key, value in arguments.items() if key in signature.parameters}
    result = model(**arguments)
    if torch.is_tensor(result):
        return {"logits": result}
    if isinstance(result, dict):
        output = dict(result)
    elif hasattr(result, "logits"):
        output = {key: getattr(result, key) for key in ("logits", "past_key_values", "hidden_states", "aux_loss")
                  if getattr(result, key, None) is not None}
    else:
        raise TypeError("causal model must return logits or an output containing logits")
    if "logits" not in output or output["logits"].ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    return output


@dataclass
class ObjectiveResult:
    loss_sum: torch.Tensor
    weight: float
    metrics: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)

    @property
    def loss(self):
        return self.loss_sum / max(self.weight, 1)


def token_log_probs(logits, labels):
    targets = labels[:, 1:]
    valid = targets != -100
    scores = logits[:, :-1].float().masked_fill(~valid.unsqueeze(-1), 0)
    log_probs = F.log_softmax(scores, dim=-1)
    selected = log_probs.gather(-1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~valid, 0), valid


def _assert_independent(model, policy):
    policy_storage = {(parameter.device, parameter.untyped_storage().data_ptr()) for parameter in policy.parameters()}
    if any((parameter.device, parameter.untyped_storage().data_ptr()) in policy_storage for parameter in model.parameters()):
        raise ValueError("reference/teacher and policy must not share parameters")


def _freeze_auxiliary(model, policy, device):
    model = unwrap_model(model)
    _assert_independent(model, policy)
    model.to(device).eval().requires_grad_(False)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        for tensor in list(model.parameters()) + list(model.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)


class CausalLMObjective:
    def normalization_weight(self, batch):
        return int((torch.as_tensor(batch["labels"])[:, 1:] != -100).sum())

    def prepare(self, policy, device):
        pass

    def config_dict(self):
        return {"type": type(self).__name__}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("unexpected objective state")

    def __call__(self, model, batch):
        output = forward_model(model, batch, use_cache=False)
        log_probs, valid = token_log_probs(output["logits"], batch["labels"])
        count = int(valid.sum())
        loss_sum = -log_probs.sum()
        auxiliary = output.get("aux_loss")
        if auxiliary is not None:
            loss_sum = loss_sum + auxiliary * count
        return ObjectiveResult(loss_sum, count, {"nll": float((-log_probs.sum()).detach()) / max(count, 1),
                                                "valid_tokens": count}, output)


class SFT(CausalLMObjective):
    """Assistant masking belongs to SFTCollator; shifting happens here once."""


class DPO(CausalLMObjective):
    def normalization_weight(self, batch):
        return len(batch["chosen_input_ids"])

    def __init__(self, reference, beta=0.1):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.reference, self.beta = reference, float(beta)

    def prepare(self, policy, device):
        _freeze_auxiliary(self.reference, policy, device)

    def config_dict(self):
        return {"type": "DPO", "beta": self.beta}

    def state_dict(self):
        return {"reference": unwrap_model(self.reference).state_dict()}

    def load_state_dict(self, state):
        unwrap_model(self.reference).load_state_dict(state["reference"])

    def __call__(self, model, batch):
        policy_logps, reference_logps, aux_losses = [], [], []
        outputs = {}
        self.reference.eval()
        for branch in ("chosen", "rejected"):
            data = {key[len(branch) + 1:]: value for key, value in batch.items() if key.startswith(branch + "_")}
            output = forward_model(model, data, use_cache=False)
            logps, valid = token_log_probs(output["logits"], data["labels"])
            if not bool(valid.any(-1).all()):
                raise ValueError("every preference branch needs at least one supervised token")
            policy_logps.append(logps.sum(-1))
            with torch.no_grad():
                reference = forward_model(self.reference, data, use_cache=False)
                ref_logps, _ = token_log_probs(reference["logits"], data["labels"])
                reference_logps.append(ref_logps.sum(-1))
            if output.get("aux_loss") is not None:
                aux_losses.append(output["aux_loss"])
            outputs[branch] = output
        margin = self.beta * ((policy_logps[0] - policy_logps[1]) - (reference_logps[0] - reference_logps[1]))
        losses = -F.logsigmoid(margin)
        if aux_losses:
            losses = losses + sum(aux_losses) / len(aux_losses)
        return ObjectiveResult(losses.sum(), len(losses),
                               {"preference_accuracy": float((margin > 0).float().mean()),
                                "reward_margin": float(margin.detach().mean())}, outputs)


class Distillation(CausalLMObjective):
    def __init__(self, teacher, temperature=2.0, alpha=0.5):
        if temperature <= 0 or not 0 <= alpha <= 1:
            raise ValueError("temperature must be positive and alpha in [0, 1]")
        self.teacher, self.temperature, self.alpha = teacher, float(temperature), float(alpha)

    def prepare(self, policy, device):
        _freeze_auxiliary(self.teacher, policy, device)

    def config_dict(self):
        return {"type": "Distillation", "temperature": self.temperature, "alpha": self.alpha}

    def state_dict(self):
        return {"teacher": unwrap_model(self.teacher).state_dict()}

    def load_state_dict(self, state):
        unwrap_model(self.teacher).load_state_dict(state["teacher"])

    def __call__(self, model, batch):
        output = forward_model(model, batch, use_cache=False)
        self.teacher.eval()
        with torch.no_grad():
            teacher = forward_model(self.teacher, batch, use_cache=False)["logits"]
        logits = output["logits"]
        if logits.shape != teacher.shape:
            raise ValueError("logit distillation requires aligned sequence positions and vocabulary")
        logps, valid = token_log_probs(logits, batch["labels"])
        temperature = self.temperature
        kl = F.kl_div(F.log_softmax(logits[:, :-1].float().masked_fill(~valid.unsqueeze(-1), 0) / temperature, -1),
                      F.softmax(teacher[:, :-1].float().masked_fill(~valid.unsqueeze(-1), 0) / temperature, -1), reduction="none").sum(-1)
        count = int(valid.sum())
        kl_sum = (kl * valid).sum() * temperature ** 2
        loss = (1 - self.alpha) * -logps.sum() + self.alpha * kl_sum
        if output.get("aux_loss") is not None:
            loss = loss + output["aux_loss"] * count
        return ObjectiveResult(loss, count, {"distillation_kl": float(kl_sum.detach()) / max(count, 1)}, output)

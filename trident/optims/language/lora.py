"""Small linear LoRA adapters with explicit, reversible merging."""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA rank/alpha/dropout")
        self.base, self.rank, self.alpha = base, int(rank), float(alpha)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)
        self.merged = False

    def forward(self, value):
        output = self.base(value)
        if not self.merged:
            output = output + ((self.dropout(value) @ self.lora_A.t()) @ self.lora_B.t()) * (self.alpha / self.rank)
        return output

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.base.weight.add_((self.lora_B.float() @ self.lora_A.float()).to(self.base.weight.dtype), alpha=self.alpha / self.rank)
            self.merged = True
        return self.base


def inject_lora(model, rank=8, alpha=16, target_modules=("q_proj", "v_proj"), dropout=0.0):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    if not target_modules:
        raise ValueError("target_modules cannot be empty")
    matches = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and any(name == target or name.endswith("." + target) for target in target_modules)]
    if not matches:
        raise ValueError("no linear modules matched LoRA targets")
    # Validate before freezing the base model.
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("invalid LoRA configuration")
    model.requires_grad_(False)
    for name, module in matches:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
    return {"rank": rank, "alpha": alpha, "target_modules": list(target_modules), "dropout": dropout}


def adapter_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key.endswith("lora_A") or key.endswith("lora_B")}


def load_adapter_state_dict(model, state):
    expected = set(adapter_state_dict(model))
    if set(state) != expected:
        raise ValueError("adapter keys do not match installed LoRA modules")
    model.load_state_dict(state, strict=False)


def merge_lora(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.merge())
    return model

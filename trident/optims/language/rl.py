"""Native group-policy and actor-critic objectives plus rollout orchestration."""
import copy
import math
from pathlib import Path

import torch
from torch import nn

from trident.optims.trainers import TrainingPlan
from .objectives import (CausalLMObjective, ObjectiveResult, _assert_independent, _freeze_auxiliary,
                         forward_model, token_log_probs, unwrap_model)
from .rollout import TorchRolloutEngine
from .runner import (_collective_stage, _configure_distributed, _distributed_scalar,
                     _load_training_checkpoint, _loader, _local_ddp_backward, _next_window,
                     _notify, _notify_ranks, _rank, _update_window, _world, move_batch,
                     save_training_checkpoint)


class GRPO(CausalLMObjective):
    def normalization_weight(self, batch):
        return len(batch["input_ids"])

    def __init__(self, reference, beta=0.04, epsilon=0.2, loss_type="grpo", epsilon_high=5.0):
        if beta < 0 or not 0 < epsilon < 1 or loss_type not in ("grpo", "cispo") or epsilon_high <= 0:
            raise ValueError("invalid GRPO/CISPO configuration")
        self.reference, self.beta, self.epsilon = reference, float(beta), float(epsilon)
        self.loss_type, self.epsilon_high = loss_type, float(epsilon_high)

    def prepare(self, policy, device):
        _freeze_auxiliary(self.reference, policy, device)

    def config_dict(self):
        return {"type": "GRPO", "beta": self.beta, "epsilon": self.epsilon,
                "loss_type": self.loss_type, "epsilon_high": self.epsilon_high}

    def state_dict(self):
        return {"reference": unwrap_model(self.reference).state_dict()}

    def load_state_dict(self, state):
        unwrap_model(self.reference).load_state_dict(state["reference"])

    def prepare_rollout(self, rollout, rewards):
        advantages = torch.zeros_like(rewards)
        degenerate = 0
        for group in rollout.group_ids.unique():
            selected = rollout.group_ids == group
            values = rewards[selected]
            if values.numel() < 2:
                raise ValueError("GRPO requires at least two completions per prompt")
            deviation = values.std(unbiased=False)
            if float(deviation) < 1e-8:
                degenerate += 1
            else:
                advantages[selected] = (values - values.mean()) / (deviation + 1e-4)
        rollout.tensors["advantages"] = advantages.detach()
        rollout.tensors["rewards"] = rewards.detach()
        return {"reward": float(rewards.mean()), "degenerate_groups": degenerate}

    def _terms(self, model, batch):
        output = forward_model(model, batch, use_cache=False)
        logps, valid = token_log_probs(output["logits"], batch["labels"])
        self.reference.eval()
        with torch.no_grad():
            reference = forward_model(self.reference, batch, use_cache=False)
            ref_logps, _ = token_log_probs(reference["logits"], batch["labels"])
        log_ratio = logps - batch["old_log_probs"].detach()
        ratio = log_ratio.exp()
        difference = ref_logps - logps
        kl = difference.exp() - difference - 1
        return output, logps, valid, ratio, kl

    def __call__(self, model, batch):
        output, logps, valid, ratio, kl = self._terms(model, batch)
        advantage = batch["advantages"].detach().unsqueeze(-1)
        if self.loss_type == "cispo":
            policy = -(ratio.clamp(max=self.epsilon_high).detach() * advantage * logps)
        else:
            policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantage)
        counts = valid.sum(-1).clamp_min(1)
        per_sequence = ((policy + self.beta * kl) * valid).sum(-1) / counts
        if output.get("aux_loss") is not None:
            per_sequence = per_sequence + output["aux_loss"]
        return ObjectiveResult(per_sequence.sum(), len(per_sequence),
                               {"kl": float(((kl * valid).sum(-1) / counts).detach().mean()),
                                "reward": float(batch["rewards"].mean())}, output)


class ValueModel(nn.Module):
    """Separate critic; the backbone must expose last-layer hidden states."""
    def __init__(self, backbone, hidden_size=None):
        super().__init__()
        self.backbone = backbone
        hidden_size = hidden_size or backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        output = forward_model(self.backbone, batch, use_cache=False, output_hidden_states=True)
        hidden = output.get("hidden_states")
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[-1]
        if hidden is None:
            raise ValueError("critic backbone must expose hidden_states")
        return self.value_head(hidden).squeeze(-1)[:, :-1].float()


class PPO(GRPO):
    def __init__(self, reference, critic, beta=0.04, epsilon=0.2, value_coef=0.5,
                 gamma=1.0, gae_lambda=0.95, critic_lr=1e-4):
        super().__init__(reference, beta, epsilon)
        if value_coef < 0 or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or critic_lr <= 0:
            raise ValueError("invalid PPO configuration")
        self.critic, self.value_coef = critic, float(value_coef)
        self.gamma, self.gae_lambda, self.critic_lr = float(gamma), float(gae_lambda), float(critic_lr)
        self.critic_optimizer = None

    def prepare(self, policy, device):
        super().prepare(policy, device)
        _assert_independent(self.critic, policy)
        _assert_independent(self.critic, self.reference)
        self.critic.to(device).requires_grad_(True)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        if _world() > 1:
            self.critic = torch.nn.parallel.DistributedDataParallel(
                self.critic, device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True)

    def optimizers(self):
        return [self.critic_optimizer]

    def config_dict(self):
        result = super().config_dict()
        result.update(type="PPO", value_coef=self.value_coef, gamma=self.gamma,
                      gae_lambda=self.gae_lambda, critic_lr=self.critic_lr)
        return result

    def state_dict(self):
        result = super().state_dict()
        result.update(critic=unwrap_model(self.critic).state_dict(), critic_optimizer=self.critic_optimizer.state_dict())
        return result

    def load_state_dict(self, state):
        super().load_state_dict(state)
        unwrap_model(self.critic).load_state_dict(state["critic"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    @torch.no_grad()
    def prepare_rollout(self, rollout, rewards):
        self.critic.eval()
        data = rollout.tensors
        values = self.critic(data)
        valid = data["labels"][:, 1:] != -100
        advantages, returns = torch.zeros_like(values), torch.zeros_like(values)
        for row in range(len(values)):
            positions = valid[row].nonzero().flatten()
            advantage, next_value = values.new_zeros(()), values.new_zeros(())
            for offset in range(len(positions) - 1, -1, -1):
                position = positions[offset]
                reward = rewards[row] if offset == len(positions) - 1 else 0.0
                delta = reward + self.gamma * next_value - values[row, position]
                advantage = delta + self.gamma * self.gae_lambda * advantage
                advantages[row, position] = advantage
                returns[row, position] = advantage + values[row, position]
                next_value = values[row, position]
        selected = advantages[valid]
        if selected.numel() > 1 and float(selected.std(unbiased=False)) > 1e-8:
            advantages = ((advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-4)) * valid
        data.update(advantages=advantages.detach(), returns=returns.detach(),
                    old_values=values.detach(), rewards=rewards.detach())
        return {"reward": float(rewards.mean())}

    def __call__(self, model, batch):
        output, logps, valid, ratio, kl = self._terms(model, batch)
        advantages = batch["advantages"].detach()
        policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantages)
        self.critic.train()
        values = self.critic(batch)
        old_values = batch["old_values"].detach()
        clipped = old_values + (values - old_values).clamp(-self.epsilon, self.epsilon)
        target = batch["returns"].detach()
        value_loss = 0.5 * torch.maximum((values - target).square(), (clipped - target).square())
        counts = valid.sum(-1).clamp_min(1)
        per_sequence = ((policy + self.beta * kl + self.value_coef * value_loss) * valid).sum(-1) / counts
        if output.get("aux_loss") is not None:
            per_sequence = per_sequence + output["aux_loss"]
        return ObjectiveResult(per_sequence.sum(), len(values),
                               {"value_loss": float((value_loss * valid).detach().sum() / valid.sum().clamp_min(1)),
                                "reward": float(batch["rewards"].mean())}, output)


class RLTrainingPlan(TrainingPlan):
    def __init__(self, name=None):
        super().__init__(name)
        self.rollout_engine = None
        self.reward = None
        self.sampling = dict(num_generations=4, max_new_tokens=32, temperature=0.8)
        self.update_epochs = 1
        self.update_batch_size = 4

    def with_rollout(self, engine=None, **sampling):
        self.rollout_engine = engine
        self.sampling.update(sampling)
        return self

    def with_reward(self, reward):
        if not callable(reward):
            raise TypeError("reward must be callable(prompt, completion, sample)")
        self.reward = reward
        return self

    def with_policy_updates(self, epochs=1, batch_size=4):
        if epochs < 1 or batch_size < 1:
            raise ValueError("update epochs and batch size must be positive")
        self.update_epochs, self.update_batch_size = int(epochs), int(batch_size)
        return self

    def execution_config(self):
        engine = self.rollout_engine
        return {"type": "RLTrainingPlan", "sampling": self.sampling,
                "update_epochs": self.update_epochs, "update_batch_size": self.update_batch_size,
                "engine": type(engine).__name__ if engine is not None else "TorchRolloutEngine",
                "max_turns": getattr(engine, "max_turns", None),
                "max_tool_calls": getattr(engine, "max_tool_calls", None)}

    def start_now(self, **kwargs):
        if len(self.training_items) != 1 or self.reward is None:
            raise ValueError("RL requires one policy item and a reward function")
        item = self.training_items.value_list[0]
        if not isinstance(item.objective, GRPO):
            raise TypeError("RL policy objective must be GRPO, CISPO or PPO")
        loader = _loader(self)
        _configure_distributed(self, item, loader)
        if not item._prepared:
            item.prepare()
        engine = self.rollout_engine or TorchRolloutEngine(item.tokenizer)
        start_epoch, offset, restored, resume_rng = 0, 0, False, None
        resume = getattr(self, "_language_resume", None)
        if resume:
            start_epoch, offset, restored, resume_rng = _load_training_checkpoint(self, item, loader, resume)
        engine.update_policy(item.model, item.optimizer_updates)
        checkpointing = getattr(self, "_language_checkpointing", None)
        limit, start_updates = getattr(self, "_language_max_updates", None), item.optimizer_updates
        try:
            _notify_ranks(self, item, "on_training_start")
            for epoch in range(start_epoch, self.num_epochs):
                self.epochs = epoch
                item.training_context["current_epoch"] = epoch
                iterator = iter(loader)
                count = offset if epoch == start_epoch else 0
                if count and not restored:
                    for _ in range(count):
                        next(iterator)
                if resume_rng is not None:
                    from .checkpoint import restore_rng
                    restore_rng(resume_rng)
                    resume_rng = None
                restored = False
                _notify_ranks(self, item, "on_epoch_start")
                def collect_rollout(samples):
                    if not isinstance(samples, (list, tuple)):
                        raise TypeError("RL loader must yield lists of prompt records; use collate=list")
                    rollout = engine.rollout(samples, **self.sampling)
                    if rollout.policy_version != item.optimizer_updates:
                        raise ValueError("rollout policy version is stale")
                    reward_samples = [dict(samples[int(group)], trajectory=rollout.trajectories[index])
                                      if rollout.trajectories is not None and isinstance(samples[int(group)], dict)
                                      else samples[int(group)] for index, group in enumerate(rollout.group_ids)]
                    with torch.no_grad():
                        rewards = torch.tensor([self.reward(prompt, completion, sample)
                                                for prompt, completion, sample in zip(rollout.prompts, rollout.completions, reward_samples)],
                                               device=item.device, dtype=torch.float32)
                    if not bool(torch.isfinite(rewards).all()):
                        raise FloatingPointError("reward function returned NaN/Inf")
                    rollout.tensors = move_batch(rollout.tensors, item.device)
                    rollout.group_ids = rollout.group_ids.to(item.device)
                    with _local_ddp_backward([getattr(item.objective, "critic", None)]):
                        metrics = item.objective.prepare_rollout(rollout, rewards)
                    return rollout, metrics

                while (batches := _next_window(iterator, 1, item)):
                    before_rollout_updates = item.optimizer_updates
                    rollout, rollout_metrics = _collective_stage(
                        item, "rollout and reward", lambda: collect_rollout(batches[0]))
                    size = len(rollout.tensors["input_ids"])
                    for _ in range(self.update_epochs):
                        minibatches = [{key: value[index:index + self.update_batch_size] for key, value in rollout.tensors.items()}
                                       for index in range(0, size, self.update_batch_size)]
                        minibatches = iter(minibatches)
                        while (window := _next_window(minibatches, item.accumulation_steps, item)):
                            _update_window(self, item, window)
                            item.history[-1].update(rollout_metrics)
                    _collective_stage(item, "rollout policy synchronization",
                                      lambda: engine.update_policy(item.model, item.optimizer_updates))
                    count += 1
                    self.steps = item.microbatches
                    stop = limit is not None and item.optimizer_updates - start_updates >= limit
                    # RL checkpoints only occur after a complete rollout/update cycle.
                    if checkpointing and (stop or item.optimizer_updates // checkpointing["every_updates"] >
                                          before_rollout_updates // checkpointing["every_updates"]):
                        save_training_checkpoint(self, item, loader, epoch, count, Path(checkpointing["directory"]) / "latest.pt")
                    terminated = _distributed_scalar(self.is_terminate, item.device, torch.distributed.ReduceOp.MAX)
                    if stop or terminated:
                        if checkpointing:
                            save_training_checkpoint(self, item, loader, epoch, count, Path(checkpointing["directory"]) / "latest.pt")
                        _notify_ranks(self, item, "on_training_end")
                        return item.history
                _notify_ranks(self, item, "on_epoch_end")
                if checkpointing:
                    save_training_checkpoint(self, item, loader, epoch + 1, 0, Path(checkpointing["directory"]) / "latest.pt")
                offset = 0
            _notify_ranks(self, item, "on_training_end")
            return item.history
        except BaseException:
            item.optimizer.zero_grad(set_to_none=True)
            try:
                _notify(self, item, "on_training_terminated")
            except Exception:
                pass  # Keep the original failure during teardown.
            raise

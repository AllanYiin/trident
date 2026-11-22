"""Optional PyTorch LLM training. Import explicitly; no network side effects."""
from trident.models.pytorch_causal_lm import CausalLM, CausalLMConfig
from .objectives import CausalLMObjective, DPO, Distillation, ObjectiveResult, SFT
from .pipeline import StageArtifact, TrainingPipeline
from .trainer import CausalLanguageModel
from .agent import AgenticRolloutEngine, ToolRegistry, parse_response
from .rl import GRPO, PPO, RLTrainingPlan, ValueModel
from .rollout import ExternalRolloutEngine, RolloutBatch, TorchRolloutEngine, WeightedReward
from .evaluation import evaluate_generation, perplexity
from .logging import ExperimentLogger
from .pretrained import load_pretrained_weights

__all__ = ["CausalLM", "CausalLMConfig", "CausalLMObjective", "CausalLanguageModel",
           "DPO", "Distillation", "ObjectiveResult", "SFT", "StageArtifact", "TrainingPipeline",
           "AgenticRolloutEngine", "ToolRegistry", "parse_response", "GRPO", "PPO", "RLTrainingPlan",
           "ValueModel", "ExternalRolloutEngine", "RolloutBatch", "TorchRolloutEngine",
           "WeightedReward", "evaluate_generation", "perplexity", "ExperimentLogger", "load_pretrained_weights"]

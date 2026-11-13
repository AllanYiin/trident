"""Small native GRPO/CISPO/PPO smoke run with a deterministic toy reward."""
import argparse
import copy
import os

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import torch

from trident.data.pipeline import ByteTokenizer, Dataset, Iterator
from trident.optims.language import (CausalLM, CausalLMConfig, CausalLanguageModel,
                                    GRPO, PPO, RLTrainingPlan, ValueModel)


def toy_reward(prompt, completion, sample):
    # Only exercises reward plumbing. This is not a useful production reward.
    return float(sum(map(ord, completion)) % 13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--algorithm", choices=("grpo", "cispo", "ppo"), default="grpo")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = ByteTokenizer()
    network = CausalLM(CausalLMConfig(vocab_size=len(tokenizer), hidden_size=32,
                                     num_hidden_layers=1, num_attention_heads=4,
                                     num_key_value_heads=2, intermediate_size=64,
                                     max_position_embeddings=128))
    reference = copy.deepcopy(network)
    objective = (PPO(reference, ValueModel(copy.deepcopy(network))) if args.algorithm == "ppo"
                 else GRPO(reference, loss_type=args.algorithm))
    item = CausalLanguageModel(network, tokenizer).with_objective(objective).with_optimizer(torch.optim.AdamW, lr=0.0001)
    plan = (RLTrainingPlan().add_training_item(item)
            .with_data_loader(Iterator(Dataset([{"prompt": "Q:"}, {"prompt": "A:"}]), collate=list))
            .with_rollout(num_generations=4, max_new_tokens=8).with_reward(toy_reward)
            .with_checkpointing(args.output_dir, every_updates=1))
    plan.start_now()
    print(args.algorithm, "updates=", item.optimizer_updates, "checkpoint=", plan.last_checkpoint)


if __name__ == "__main__":
    main()

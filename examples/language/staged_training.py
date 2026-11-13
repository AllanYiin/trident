"""Offline Pretrain -> LoRA SFT -> DPO, with explicit stage artifacts."""
import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import torch

from trident.data.pipeline import (ByteTokenizer, CausalLMCollator, Dataset, Iterator,
                                   PreferenceCollator, SFTCollator, load_tokenizer)
from trident.optims.language import (CausalLM, CausalLMConfig, CausalLanguageModel,
                                    DPO, SFT, TrainingPipeline)
from trident.optims.trainers import TrainingPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)

    def plan(item, samples, collator):
        return (TrainingPlan().add_training_item(item)
                .with_data_loader(Iterator(Dataset(samples), batch_size=2, collate=collator)))

    def pretrain(previous):
        tokenizer = ByteTokenizer()
        network = CausalLM(CausalLMConfig(vocab_size=len(tokenizer), hidden_size=32,
                                         num_hidden_layers=1, num_attention_heads=4,
                                         num_key_value_heads=2, intermediate_size=64,
                                         max_position_embeddings=256))
        item = CausalLanguageModel(network, tokenizer).with_optimizer(torch.optim.AdamW, lr=0.001)
        return plan(item, [{"text": "two plus three equals five"}] * 4, CausalLMCollator(tokenizer))

    def sft(previous):
        tokenizer = load_tokenizer(previous.model_directory)
        network = CausalLM.from_pretrained(previous.model_directory)
        item = (CausalLanguageModel(network, tokenizer).with_objective(SFT())
                .with_lora(rank=4, alpha=8).with_optimizer(torch.optim.AdamW, lr=0.001))
        samples = [{"messages": [{"role": "user", "content": "2+3?"}, {"role": "assistant", "content": "5"}]}] * 4
        return plan(item, samples, SFTCollator(tokenizer, max_length=128))

    def dpo(previous):
        tokenizer = load_tokenizer(previous.model_directory)
        network = CausalLM.from_pretrained(previous.model_directory)
        item = (CausalLanguageModel(network, tokenizer).with_objective(DPO(copy.deepcopy(network)))
                .with_lora(rank=4, alpha=8).with_optimizer(torch.optim.AdamW, lr=0.001))
        return plan(item, [{"prompt": "2+3?", "chosen": "5", "rejected": "6"}] * 4,
                    PreferenceCollator(tokenizer, max_length=128))

    pipeline = (TrainingPipeline(Path(args.output_dir)).add_stage("pretrain", pretrain)
                .add_stage("sft", sft).add_stage("dpo", dpo))
    for artifact in pipeline.start_now(resume=args.resume):
        print(artifact.name, artifact.model_directory)


if __name__ == "__main__":
    main()

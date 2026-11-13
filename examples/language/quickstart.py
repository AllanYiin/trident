"""Offline tiny-model training; demonstrates mechanics, not chatbot quality."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

import torch

from trident.data.pipeline import ByteTokenizer, CausalLMCollator, Dataset, Iterator
from trident.optims.language import CausalLM, CausalLMConfig, CausalLanguageModel, perplexity
from trident.optims.trainers import TrainingPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--steps", default=30, type=int)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    rows = [{"text": text} for text in ("red green blue", "one two three", "hello trident", "small language model")]
    train = Iterator(Dataset(rows * (args.steps * 2)), batch_size=4,
                     collate=CausalLMCollator(tokenizer, max_length=64))
    valid = Iterator(Dataset(rows), batch_size=4, collate=CausalLMCollator(tokenizer, max_length=64))
    network = CausalLM(CausalLMConfig(vocab_size=len(tokenizer), hidden_size=32,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, intermediate_size=64,
                                     max_position_embeddings=128)).to(args.device)
    item = (CausalLanguageModel(network, tokenizer).with_precision(args.precision)
            .with_optimizer(torch.optim.AdamW, lr=0.005).with_accumulate_grads(2)
            .with_grad_clipping(1.0))
    before = perplexity(item, valid)
    plan = (TrainingPlan().add_training_item(item).with_data_loader(train)
            .with_checkpointing(output / "checkpoints", every_updates=10).limit_updates(args.steps))
    plan.start_now()
    after = perplexity(item, valid)
    item.export(output / "model")
    prompt = torch.tensor([tokenizer.encode("red ")], device=args.device)
    generated = item.generate(prompt, max_new_tokens=12)
    result = {"before_loss": before["loss"], "after_loss": after["loss"],
              "optimizer_updates": item.optimizer_updates, "valid_tokens": item.valid_tokens,
              "tokens_per_second_last_window": item.history[-1]["tokens_per_second"],
              "peak_memory_bytes": item.history[-1].get("peak_memory_bytes", 0),
              "generated_sample": tokenizer.decode(generated[0].tolist(), skip_special_tokens=True),
              "checkpoint": plan.last_checkpoint, "device": args.device, "precision": args.precision}
    if not after["loss"] < before["loss"]:
        raise RuntimeError("tiny-corpus training did not lower validation loss")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

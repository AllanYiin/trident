"""Serve a local native export; starts only when explicitly executed."""
import argparse
import os

os.environ.setdefault("TRIDENT_BACKEND", "pytorch")

from trident.data.pipeline import load_tokenizer
from trident.optims.language import CausalLM, CausalLanguageModel
from trident.optims.language.serving import create_chat_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8998)
    args = parser.parse_args()
    import uvicorn
    item = CausalLanguageModel(CausalLM.from_pretrained(args.model_dir), load_tokenizer(args.model_dir)).to(args.device)
    uvicorn.run(create_chat_app(item), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

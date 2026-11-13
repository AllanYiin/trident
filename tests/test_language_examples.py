import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,arguments", [
    ("quickstart.py", ["--steps", "20"]),
    ("staged_training.py", []),
    ("online_rl.py", ["--algorithm", "grpo"]),
    ("online_rl.py", ["--algorithm", "cispo"]),
    ("online_rl.py", ["--algorithm", "ppo"]),
])
def test_examples_run_in_requested_output_directory(tmp_path, script, arguments):
    output = tmp_path / "output"
    environment = dict(os.environ, TRIDENT_BACKEND="pytorch", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "language" / script),
                             "--output-dir", str(output)] + arguments,
                            cwd=tmp_path, env=environment, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()
    if script == "quickstart.py":
        summary = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert summary["after_loss"] < summary["before_loss"]
        assert (output / "model" / "model.pt").exists()
    if script == "staged_training.py":
        manifest = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["completed"] == ["pretrain", "sft", "dpo"]


def test_native_import_does_not_require_huggingface(tmp_path):
    environment = dict(os.environ, TRIDENT_BACKEND="pytorch", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    program = """
import importlib.abc, sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('transformers', 'tokenizers', 'safetensors', 'fastapi'):
            raise ImportError('optional package intentionally unavailable')
sys.meta_path.insert(0, BlockExternal())
from trident.data.pipeline import ByteTokenizer, CausalLMCollator
from trident.optims.language import CausalLM, CausalLMConfig
assert len(ByteTokenizer()) == 272
network = CausalLM(CausalLMConfig(vocab_size=272, hidden_size=16, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, intermediate_size=32))
print('native import without HF extras passed')
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr

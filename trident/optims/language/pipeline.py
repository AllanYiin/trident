"""Sequential stage factories; optimizer state is never implicitly inherited."""
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageArtifact:
    name: str
    directory: Path
    checkpoint: Path
    model_directory: Path


class TrainingPipeline:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stages = []
        self.artifacts = []

    def add_stage(self, name, factory):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("stage names must be simple path-safe identifiers")
        if any(stage_name == name for stage_name, _ in self.stages):
            raise ValueError("duplicate stage name")
        if not callable(factory):
            raise TypeError("stage factory must be callable")
        self.stages.append((name, factory))
        return self

    def _write_manifest(self, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / ("pipeline." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(self.directory / "pipeline.json"))
        finally:
            if temporary.exists():
                temporary.unlink()

    def start_now(self, resume=False):
        import torch.distributed as dist
        if (dist.is_initialized() and dist.get_world_size() > 1) or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("TrainingPipeline is single-process; launch DDP stages separately")
        if not self.stages:
            raise ValueError("pipeline has no stages")
        manifest_path = self.directory / "pipeline.json"
        if manifest_path.exists() and not resume:
            raise FileExistsError("pipeline already exists; use resume=True or a new directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if resume and manifest_path.exists() else {"version": 1, "stages": [name for name, _ in self.stages], "completed": []}
        if manifest["stages"] != [name for name, _ in self.stages]:
            raise ValueError("pipeline stage order changed")
        self.artifacts = []
        previous = None
        for name, factory in self.stages:
            directory = self.directory / name
            artifact = StageArtifact(name, directory, directory / "checkpoints" / "latest.pt", directory / "model")
            if name not in manifest["completed"]:
                plan = factory(previous)
                if getattr(plan, "_language_distributed", None):
                    raise ValueError("launch distributed stages separately from TrainingPipeline")
                if len(plan.training_items) != 1:
                    raise ValueError("stage factory must return a single-item language plan")
                plan.with_checkpointing(directory / "checkpoints")
                if resume and artifact.checkpoint.exists():
                    plan.resume_from(artifact.checkpoint)
                manifest.update(active=name, status="running")
                self._write_manifest(manifest)
                try:
                    plan.start_now()
                    if plan.is_terminate or getattr(plan, "_language_max_updates", None) is not None:
                        manifest["status"] = "paused"
                        self._write_manifest(manifest)
                        return self.artifacts
                    item = plan.training_items.value_list[0]
                    format = "native" if hasattr(item.model, "config") and type(item.model).__module__.startswith("trident.") else "huggingface"
                    item.export(artifact.model_directory, format=format)
                except BaseException:
                    manifest["status"] = "failed"
                    self._write_manifest(manifest)
                    raise
                manifest["completed"].append(name)
                manifest["status"] = "ready"
                self._write_manifest(manifest)
            if not artifact.checkpoint.exists() or not artifact.model_directory.exists():
                raise FileNotFoundError("completed stage artifacts are missing: " + name)
            previous = artifact
            self.artifacts.append(artifact)
        manifest.update(status="complete", active=None)
        self._write_manifest(manifest)
        return self.artifacts

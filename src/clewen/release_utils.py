"""file-level release validation; no GPU or model import required"""

import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_file(root, relative):
    # validate the manifest name, not the symlink target. HF snapshot files
    # legitimately point outside snapshots/<commit>/ into the repo's blobs/
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError(f"Invalid release path: {relative}")
    name = Path(relative)
    if name.is_absolute() or name.drive or ".." in name.parts or not name.parts:
        raise ValueError(f"Invalid release path: {relative}")
    root = Path(root).resolve()
    return root / name


def validate_release(directory, verify_checksums=False):
    root = Path(directory)
    marker = root / "release_manifest.json"
    if not marker.is_file():
        raise ValueError("Incomplete release: release_manifest.json is missing. Build the weights first.")
    inventory = json.loads(marker.read_text())
    if inventory.get("format_version") != 2:
        raise ValueError("Unsupported Clewen release format")
    for relative, info in inventory["files"].items():
        path = release_file(root, relative)
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or incomplete release file: {relative}")
        # always verify metadata/code. Hashing all ~21 GB is optional at load time
        if verify_checksums or path.suffix != ".safetensors":
            if sha256_file(path) != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {relative}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "clewen" or config.get("modes") != ["text", "decision"]:
        raise ValueError("Not a two-mode Clewen release")
    mappings = config.get("auto_map", {})
    if (
        mappings.get("AutoConfig") != "configuration_clewen.ClewenConfig"
        or mappings.get("AutoModel") != "modeling_clewen.ClewenModel"
    ):
        raise ValueError("Missing HF AutoConfig/AutoModel mappings")
    required = {
        "config.json",
        "configuration_clewen.py",
        "modeling_clewen.py",
        "adapter_runtime.py",
        "joint_schema_model.py",
        "release_utils.py",
        "qwen/config.json",
        "qwen/model.safetensors.index.json",
        "qwen/tokenizer_config.json",
        "qwen/tokenizer.json",
        "adapters/manifest.json",
        "adapters/packed.safetensors",
        "head/joint_head_config.json",
        "head/joint_head.safetensors",
    }
    index = json.loads((root / "qwen/model.safetensors.index.json").read_text())
    required.update("qwen/" + name for name in set(index["weight_map"].values()))
    if not required.issubset(inventory["files"]):
        raise ValueError(f"Incomplete inventory: {sorted(required - inventory['files'].keys())}")
    manifest = json.loads((root / "adapters/manifest.json").read_text())
    if manifest["models"] != config["source_models"]:
        raise ValueError("Backbone and adapters have different source revisions")
    return config, inventory


def copy_release(source, destination):
    """copy original checkpoint files, preserving untouched Qwen and its MTP tensors"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        validate_release(source)
        return str(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Save to a separate directory outside the source release")
    _, inventory = validate_release(source)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Destination must be empty")
    for relative in inventory["files"]:
        target = release_file(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release_file(source, relative), target)
    # completion marker is written last
    shutil.copyfile(source / "release_manifest.json", destination / "release_manifest.json")
    validate_release(destination, verify_checksums=True)
    return str(destination)

"""assemble a complete Clewen repository from existing, pinned source artifacts"""

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

MODEL_SPECS = json.loads(Path(__file__).with_name("models.json").read_text())
SOURCE_MODELS = MODEL_SPECS["clewen-flash"]["sources"]
CODE_FILES = {
    name: "src/clewen/" + name
    for name in (
        "configuration_clewen.py",
        "modeling_clewen.py",
        "adapter_runtime.py",
        "release_utils.py",
        "joint_schema_model.py",
    )
}
CODE_FILES.update(
    {
        name: "assets/" + name
        for name in (
            "requirements.txt",
            "requirements-kernels.txt",
            "LICENSE",
            "LICENSE-CLEF",
            "LICENSE-QWEN",
            "NOTICE",
            ".gitattributes",
        )
    }
)


def build_release(qwen, clef, adapters, project_root, output, variant="clewen-flash"):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    qwen, clef, adapters, project_root, output = map(Path, (qwen, clef, adapters, project_root, output))
    sys.path.insert(0, str(project_root / "src"))
    from clewen.release_utils import sha256_file, validate_release

    manifest = json.loads((adapters / "manifest.json").read_text())
    spec = MODEL_SPECS[variant]
    sources = spec["sources"]
    if manifest["models"] != sources:
        raise ValueError("Adapters do not belong to the pinned Qwen/Clef revisions")
    if not manifest.get("entries") or len({e["tensor"] for e in manifest["entries"]}) != len(manifest["entries"]):
        raise ValueError("Incomplete or duplicate adapter entries")
    if output.exists():
        saved_config, inventory = validate_release(output, verify_checksums=True)
        if saved_config["source_models"] != sources or saved_config["adapter_artifact_id"] != manifest["artifact_id"]:
            raise FileExistsError("Existing release belongs to another model or adapter artifact")
        for name, source in CODE_FILES.items():
            if sha256_file(project_root / source) != inventory["files"][name]["sha256"]:
                raise FileExistsError("Existing release has different code; choose a new release ID")
        return {"directory": str(output), "cached": True, "bytes": sum(x["bytes"] for x in inventory["files"].values())}
    index = json.loads((qwen / "model.safetensors.index.json").read_text())
    mtp_keys = [key for key in index["weight_map"] if key.startswith("mtp.")]
    if not mtp_keys:
        raise ValueError("Pinned Qwen checkpoint must retain its MTP tensors")
    missing = [name for name in set(index["weight_map"].values()) if not (qwen / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Qwen shards: {missing}")
    if (clef / "joint_schema_model.py").exists() and sha256_file(clef / "joint_schema_model.py") != sha256_file(
        project_root / "src/clewen/joint_schema_model.py"
    ):
        raise ValueError("Vendored Clef implementation differs from the pinned source")

    work = output.with_name(output.name + ".building-" + uuid.uuid4().hex[:8])
    work.mkdir(parents=True)

    def copy(source, relative):
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    try:
        for name, source in CODE_FILES.items():
            copy(project_root / source, name)
        # copy the original shards/index; never serialize installed wrapper modules
        # all MTP tensors remain byte-for-byte in those shards
        for path in sorted(qwen.iterdir()):
            if path.is_file() and path.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken", ".safetensors"}:
                copy(path, "qwen/" + path.name)
        for name in ["joint_head_config.json", "joint_head.safetensors"]:
            copy(clef / name, "head/" + name)
        processor_files = {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
        }
        for name in processor_files:
            if (qwen / name).is_file():
                copy(qwen / name, name)
        copy(adapters / "manifest.json", "adapters/manifest.json")
        packed_path = adapters / "packed.safetensors"
        keys = {"low_rank": ("A", "B"), "dense_delta": ("delta",), "parameter_override": ("target",)}
        expected = {e["file"] + "/" + k for e in manifest["entries"] for k in keys[e["kind"]]}
        if packed_path.is_file():
            with safe_open(str(packed_path), framework="pt") as packed:
                if set(packed.keys()) != expected:
                    raise ValueError("Packed adapter keys mismatch")
                for entry in manifest["entries"]:
                    original = load_file(str(adapters / entry["file"]))
                    if set(original) != set(keys[entry["kind"]]):
                        raise ValueError("Unexpected original adapter keys")
                    for key, tensor in original.items():
                        if not torch.equal(packed.get_tensor(entry["file"] + "/" + key), tensor):
                            raise ValueError(f"Packing changed {entry['tensor']}:{key}")
            copy(packed_path, "adapters/packed.safetensors")
        else:
            packed = {}
            for entry in manifest["entries"]:
                original = load_file(str(adapters / entry["file"]))
                if set(original) != set(keys[entry["kind"]]):
                    raise ValueError("Unexpected adapter keys")
                packed.update({entry["file"] + "/" + key: value for key, value in original.items()})
            save_file(packed, str(work / "adapters/packed.safetensors"))
            with safe_open(str(work / "adapters/packed.safetensors"), framework="pt") as saved:
                if set(saved.keys()) != expected or any(
                    not torch.equal(saved.get_tensor(k), v) for k, v in packed.items()
                ):
                    raise ValueError("Packed save did not preserve adapter tensors")
            del packed
        config = {
            "model_type": "clewen",
            "architectures": ["ClewenModel"],
            "format_version": 2,
            "auto_map": {
                "AutoConfig": "configuration_clewen.ClewenConfig",
                "AutoModel": "modeling_clewen.ClewenModel",
                "AutoModelForCausalLM": "modeling_clewen.ClewenModel",
            },
            "name": spec["name"],
            "variant": variant,
            "modes": ["text", "decision"],
            "dtype": "bfloat16",
            "source_models": sources,
            "adapter_artifact_id": manifest["artifact_id"],
            "default_max_input_tokens": 4096,
            "mtp_preserved": True,
            "mtp_enabled": False,
            "inference_only": True,
        }
        (work / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print("Hashing and validating the assembled release...", flush=True)
        inventory = {"format_version": 2, "mtp_tensor_count": len(mtp_keys), "files": {}}
        for path in sorted(work.rglob("*")):
            if path.is_file():
                inventory["files"][path.relative_to(work).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        # completion marker appears only after all source files have been copied
        (work / "release_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
        validate_release(work)
        work.rename(output)
        return {
            "directory": str(output),
            "cached": False,
            "mtp_tensor_count": len(mtp_keys),
            "adapter_tensor_count": len(expected),
            "files": len(inventory["files"]),
            "bytes": sum(x["bytes"] for x in inventory["files"].values()),
        }
    except Exception:
        shutil.rmtree(work)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ["qwen", "clef", "adapters", "project-root", "output"]:
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--variant", choices=MODEL_SPECS, default="clewen-flash")
    print(json.dumps(build_release(**vars(parser.parse_args())), indent=2))

"""download pinned checkpoints and audit every common tensor"""

import json
import time
from pathlib import Path


def prepare_sources(sources, folder, checkpoint=lambda: None):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    start = time.monotonic()
    torch.set_num_threads(2)
    paths = {}
    for key, spec in sources.items():
        print("downloading " + spec["repo_id"], flush=True)
        paths[key] = Path(snapshot_download(**spec, max_workers=4, ignore_patterns=["*.md", "*.png", "*.jpg"]))
        checkpoint()
    indexes = {
        key: json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        for key, path in paths.items()
    }
    if set(indexes["clef"]) - set(indexes["qwen"]):
        raise ValueError("Clef contains tensors absent from the base Qwen checkpoint")
    handles = {
        key: {name: safe_open(str(paths[key] / name), framework="pt") for name in set(index.values())}
        for key, index in indexes.items()
    }
    entries = []
    for index, name in enumerate(sorted(indexes["clef"])):
        target = handles["clef"][indexes["clef"][name]].get_tensor(name)
        base = handles["qwen"][indexes["qwen"][name]].get_tensor(name)
        if target.shape != base.shape:
            raise ValueError(f"tensor shape mismatch: {name}")
        equal = all(
            torch.equal(target.reshape(-1)[offset : offset + 8_388_608], base.reshape(-1)[offset : offset + 8_388_608])
            for offset in range(0, target.numel(), 8_388_608)
        )
        entries.append(
            {
                "tensor": name,
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "identical": equal,
                "elements": target.numel(),
            }
        )
        del target, base
        if (index + 1) % 128 == 0:
            print(f"audited {index + 1} tensors", flush=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "models": sources,
        "entries": entries,
        "qwen_only": sorted(set(indexes["qwen"]) - set(indexes["clef"])),
        "elapsed_seconds": time.monotonic() - start,
    }
    (folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    checkpoint()
    return {
        "seconds": report["elapsed_seconds"],
        "tensors": len(entries),
        "changed": sum(not entry["identical"] for entry in entries),
    }

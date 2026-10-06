"""recover switchable deltas from pinned Qwen and Clef checkpoints"""

import json
import time
from pathlib import Path


def recover_adapters(specs, paths, folder, budget_seconds=3600, checkpoint=lambda: None):
    import gc
    import hashlib

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    start = time.monotonic()
    folder = Path(folder)
    artifact_id = folder.name

    def emit(row):
        print(json.dumps(row), flush=True)
        return row

    audit_data = json.loads((folder / "audit.json").read_text())
    assert audit_data["models"] == specs
    indexes = {
        key: json.loads((p / "model.safetensors.index.json").read_text())["weight_map"] for key, p in paths.items()
    }
    handles = {
        key: {file: safe_open(str(paths[key] / file), framework="pt", device="cpu") for file in set(index.values())}
        for key, index in indexes.items()
    }
    emit(
        {
            "kind": "environment",
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "budget_seconds": budget_seconds,
        }
    )
    changed = [e for e in audit_data["entries"] if not e["identical"]]
    changed = sorted(changed, key=lambda entry: entry["tensor"])
    for entry in changed:
        name, shape = entry["tensor"], entry["shape"]
        filename = hashlib.sha256(name.encode()).hexdigest()[:20] + ".safetensors"
        metadata_file = folder / (filename + ".json")
        if metadata_file.exists() and (folder / filename).exists():
            saved_info = json.loads(metadata_file.read_text())
            emit({**saved_info, "adapter_kind": saved_info["kind"], "kind": "recovered", "cached": True})
            continue
        if time.monotonic() - start >= budget_seconds - 30:
            emit({"kind": "budget_stop", "stage": "recover", "elapsed_seconds": time.monotonic() - start})
            checkpoint()
            raise TimeoutError("adapter recovery budget exhausted; saved tensors can be resumed")
        tensor_start = time.monotonic()
        a_cpu = handles["clef"][indexes["clef"][name]].get_tensor(name)
        b_cpu = handles["qwen"][indexes["qwen"][name]].get_tensor(name)
        is_linear = len(shape) == 2 and name.startswith("model.language_model.layers.") and name.endswith(".weight")
        if not is_linear:
            # do not silently discard changes outside conventional LoRA matrices
            if entry["elements"] * 2 > 64 * 2**20:
                raise RuntimeError(f"Unexpected large non-linear change requires review: {name}, {shape}")
            kind, tensors, rank, residual, energy, method = (
                "parameter_override",
                {"target": a_cpu.clone()},
                None,
                0.0,
                1.0,
                "exact_replacement",
            )
        else:
            a = a_cpu.to(device="cuda", dtype=torch.float32)
            b = b_cpu.to(device="cuda", dtype=torch.float32)
            delta = a - b
            total_energy = float(delta.double().square().sum().item())
            target_rank = min(256, *shape)
            if target_rank * sum(shape) >= entry["elements"]:
                kind, tensors, rank = "dense_delta", {"delta": delta.to(torch.bfloat16).cpu().contiguous()}, None
                reconstruction = tensors["delta"].to(device="cuda", dtype=torch.float32)
                method = "dense_small_matrix"
            else:
                torch.manual_seed(1729 + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16))
                q = min(target_rank + 64, *shape)
                u, s, v = torch.svd_lowrank(delta, q=q, niter=2)
                root_s = s[:target_rank].sqrt()
                A = (v[:, :target_rank].T * root_s[:, None]).to(torch.bfloat16)
                B = (u[:, :target_rank] * root_s[None, :]).to(torch.bfloat16)
                reconstruction = B.float() @ A.float()
                preliminary = float((delta - reconstruction).double().square().sum().item()) / total_energy
                if preliminary > 0.0025:
                    # a slow singular-value tail can require more oversampling/iterations
                    u, s, v = torch.svd_lowrank(delta, q=min(target_rank + 256, *shape), niter=4)
                    root_s = s[:target_rank].sqrt()
                    A = (v[:, :target_rank].T * root_s[:, None]).to(torch.bfloat16)
                    B = (u[:, :target_rank] * root_s[None, :]).to(torch.bfloat16)
                    reconstruction = B.float() @ A.float()
                    method = "randomized_svd_q512_niter4"
                else:
                    method = "randomized_svd_q320_niter2"
                kind, tensors, rank = "low_rank", {"A": A.cpu().contiguous(), "B": B.cpu().contiguous()}, target_rank
            error_energy = float((delta - reconstruction).double().square().sum().item())
            residual = (error_energy / total_energy) ** 0.5
            energy = 1.0 - error_energy / total_energy
            del a, b, delta, reconstruction
            if kind == "low_rank":
                del u, s, v, A, B, root_s
        save_file(tensors, str(folder / filename))
        info = {
            "tensor": name,
            "shape": shape,
            "kind": kind,
            "file": filename,
            "rank": rank,
            "relative_delta_residual": residual,
            "delta_energy_retained": energy,
            "method": method,
            "seconds": time.monotonic() - tensor_start,
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
        }
        metadata_file.write_text(json.dumps(info, indent=2) + "\n")
        checkpoint()
        emit({**info, "adapter_kind": kind, "kind": "recovered", "cached": False})
        del tensors, a_cpu, b_cpu
        gc.collect()
    all_changed = [e for e in audit_data["entries"] if not e["identical"]]
    recovered = {
        json.loads(p.read_text())["tensor"]: json.loads(p.read_text()) for p in folder.glob("*.safetensors.json")
    }
    missing = [e["tensor"] for e in all_changed if e["tensor"] not in recovered]
    if not missing:
        manifest = {
            "models": specs,
            "artifact_id": artifact_id,
            "rank": 256,
            "method": "float32_delta_randomized_svd_bf16_factors",
            "entries": [recovered[e["tensor"]] for e in all_changed],
            "unchanged_tensors": sum(e["identical"] for e in audit_data["entries"]),
            "qwen_only": audit_data["qwen_only"],
            "adapter_bytes": sum(e["bytes"] for e in recovered.values()),
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checkpoint()
    if missing:
        raise RuntimeError(f"Missing recovered tensors: {missing}")
    return {
        "adapter_bytes": manifest["adapter_bytes"],
        "tensors": len(manifest["entries"]),
        "elapsed_seconds": time.monotonic() - start,
    }

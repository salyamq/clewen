"""switchable recovered deltas; inactive mode leaves base weights untouched"""

from pathlib import Path


def install_adapters(model, artifact_dir, manifest, packed_tensors=None):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    class DeltaLinear(torch.nn.Module):
        def __init__(self, base, tensors, kind):
            super().__init__()
            self.base = base
            self.enabled = False
            self.kind = kind
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor.to(device=base.weight.device, dtype=base.weight.dtype))

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            result = self.base(x)
            if not self.enabled:
                return result
            if self.kind == "low_rank":
                update = F.linear(F.linear(x, self.A), self.B)
            else:
                update = F.linear(x, self.delta)
            return result + update

    # validate every target before replacing any modules
    kinds = {"low_rank": ("A", "B"), "dense_delta": ("delta",), "parameter_override": ("target",)}
    entries = manifest["entries"]
    if len({entry["tensor"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate adapter targets")
    if packed_tensors is not None:
        expected = {entry["file"] + "/" + key for entry in entries for key in kinds[entry["kind"]]}
        if set(packed_tensors) != expected:
            raise ValueError("Packed adapter keys do not match the manifest")
    for entry in entries:
        kind, name = entry["kind"], entry["tensor"]
        if kind not in kinds:
            raise ValueError(f"Unknown adapter kind: {kind}")
        parameter = model.get_parameter(name)
        if list(parameter.shape) != entry["shape"]:
            raise ValueError(f"Adapter shape mismatch at {name}")
        tensors = (
            load_file(str(Path(artifact_dir) / entry["file"]))
            if packed_tensors is None
            else {key: packed_tensors[entry["file"] + "/" + key] for key in kinds[kind]}
        )
        if kind == "low_rank":
            A, B = tensors["A"], tensors["B"]
            if (
                A.ndim != 2
                or B.ndim != 2
                or A.shape[0] != B.shape[1]
                or (B.shape[0], A.shape[1]) != tuple(parameter.shape)
            ):
                raise ValueError(f"Invalid low-rank factors at {name}")
        elif tensors[kinds[kind][0]].shape != parameter.shape:
            raise ValueError(f"Invalid adapter tensor at {name}")
        if kind in {"low_rank", "dense_delta"} and not isinstance(
            model.get_submodule(name.removesuffix(".weight")), torch.nn.Linear
        ):
            raise TypeError(f"Expected Linear at {name}")

    wrappers, replacements = [], []
    for entry in manifest["entries"]:
        name = entry["tensor"]
        if packed_tensors is None:
            tensors = load_file(str(Path(artifact_dir) / entry["file"]), device="cpu")
        else:
            keys = {"low_rank": ("A", "B"), "dense_delta": ("delta",), "parameter_override": ("target",)}[entry["kind"]]
            tensors = {key: packed_tensors[entry["file"] + "/" + key] for key in keys}
        if entry["kind"] in {"low_rank", "dense_delta"}:
            module_name = name.removesuffix(".weight")
            base = model.get_submodule(module_name)
            if not isinstance(base, torch.nn.Linear):
                raise TypeError(f"Expected Linear at {module_name}, got {type(base).__name__}")
            parent_name, _, child = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapper = DeltaLinear(base, tensors, entry["kind"])
            setattr(parent, child, wrapper)
            wrappers.append(wrapper)
        else:
            module_name, _, parameter_name = name.rpartition(".")
            owner = model.get_submodule(module_name) if module_name else model
            original = owner._parameters[parameter_name]
            target = torch.nn.Parameter(tensors["target"].to(original.device, original.dtype), requires_grad=False)
            replacements.append((owner, parameter_name, original, target))

    class Controller(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = False
            self.originals = torch.nn.ParameterList([item[2] for item in replacements])
            self.targets = torch.nn.ParameterList([item[3] for item in replacements])

        def set_enabled(self, enabled):
            for wrapper in wrappers:
                wrapper.enabled = bool(enabled)
            for index, (owner, name, _, _) in enumerate(replacements):
                owner._parameters[name] = self.targets[index] if enabled else self.originals[index]
            self.enabled = bool(enabled)

        def parameter_bytes(self):
            return sum(t.numel() * t.element_size() for wrapper in wrappers for t in wrapper.buffers()) + sum(
                target.numel() * target.element_size() for target in self.targets
            )

    return Controller()

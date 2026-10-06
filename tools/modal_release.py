"""build/test the Clewen releases from existing Modal artifacts. Does not publish"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("clewen-releases")
volume = modal.Volume.from_name("clef-qwen-evaluation", create_if_missing=True)
BASE_PACKAGES = (
    "transformers==5.10.2",
    "accelerate>=1.12,<2",
    "huggingface-hub>=1.5,<2",
    "safetensors>=0.6,<1",
    "pillow>=11,<13",
    "numpy>=2,<3",
    "fastapi>=0.115,<1",
    "uvicorn>=0.34,<1",
    "httpx>=0.28,<1",
)
CONV_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
    "causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)


def project_files(runtime):
    return (
        runtime.env({"HF_HOME": "/data/hf", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
        .add_local_dir(ROOT / "src", "/opt/project/src", ignore=["__pycache__", "*.pyc"])
        .add_local_dir(ROOT / "tools", "/opt/project/tools", ignore=["__pycache__", "*.pyc"])
        .add_local_dir(ROOT / "tests", "/opt/project/tests", ignore=["__pycache__", "*.pyc"])
        .add_local_dir(ROOT / "assets", "/opt/project/assets")
    )


runtime = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(*BASE_PACKAGES)
)
image = project_files(runtime)
# official prebuilt extension, matched to Torch 2.10, CUDA 12 and Python 3.11
# no nvcc/compiler or source build is required
gpu_image = project_files(
    runtime.pip_install("flash-linear-attention[cuda]==0.5.2")
    .pip_install(CONV_WHEEL)
    .run_commands("python -c 'import causal_conv1d, fla; print(causal_conv1d.__version__)'")
)


def release_path(release_id):
    if not release_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in release_id
    ):
        raise ValueError("release_id must contain only letters, digits, '-' and '_'")
    return Path("/data/releases") / release_id


def source_paths(spec):
    return {
        key: Path("/data/hf/hub") / ("models--" + item["repo_id"].replace("/", "--")) / "snapshots" / item["revision"]
        for key, item in spec["sources"].items()
    }


@app.function(
    image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=3600, max_containers=1, scaledown_window=2
)
def prepare(variant):
    import sys

    sys.path.insert(0, "/opt/project/tools")
    from build_release import MODEL_SPECS
    from prepare_sources import prepare_sources

    spec = MODEL_SPECS[variant]
    return prepare_sources(spec["sources"], Path("/data/adapters") / spec["adapter_id"], volume.commit)


@app.function(
    image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=1200, max_containers=1, scaledown_window=2
)
def build(variant, release_id):
    import sys

    sys.path.insert(0, "/opt/project/tools")
    from build_release import MODEL_SPECS, build_release

    spec = MODEL_SPECS[variant]
    paths = source_paths(spec)
    result = build_release(
        paths["qwen"],
        paths["clef"],
        Path("/data/adapters") / spec["adapter_id"],
        Path("/opt/project"),
        release_path(release_id),
        variant=variant,
    )
    volume.commit()
    return result


@app.function(
    image=gpu_image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=2,
    memory=16384,
    timeout=3700,
    max_containers=1,
    scaledown_window=2,
)
def recover(variant, budget_seconds=3600):
    import sys

    sys.path.insert(0, "/opt/project/tools")
    from build_release import MODEL_SPECS
    from recover_adapters import recover_adapters

    spec = MODEL_SPECS[variant]
    return recover_adapters(
        spec["sources"], source_paths(spec), Path("/data/adapters") / spec["adapter_id"], budget_seconds, volume.commit
    )


@app.function(image=image, cpu=1, memory=4096, timeout=180, max_containers=1, scaledown_window=2)
def unit_tests():
    import sys
    import unittest

    sys.path[:0] = ["/opt/project/src", "/opt/project/tools"]
    suite = unittest.defaultTestLoader.discover("/opt/project/tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("CPU unit tests failed")
    return {"tests": result.testsRun, "passed": True}


def validate_on_gpu(variant, release_id):
    import sys

    sys.path[:0] = ["/opt/project/src", "/opt/project/tools", "/opt/project/tests"]
    from build_release import MODEL_SPECS
    from gpu_check import check
    from clewen.release_utils import sha256_file

    spec = MODEL_SPECS[variant]
    result = check(
        release_path(release_id),
        source_paths(spec)["clef"],
        variant,
        Path("/data/test_media"),
    )
    result["release_manifest_sha256"] = sha256_file(release_path(release_id) / "release_manifest.json")
    folder = Path("/data/release_checks") / release_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gpu_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    volume.commit()
    return result


@app.function(
    image=gpu_image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=2,
    memory=32768,
    timeout=600,
    startup_timeout=900,
    max_containers=1,
    scaledown_window=2,
)
def gpu_check(variant, release_id):
    return validate_on_gpu(variant, release_id)


@app.function(
    image=gpu_image,
    volumes={"/data": volume},
    gpu="H200",
    cpu=2,
    memory=32768,
    timeout=900,
    startup_timeout=900,
    max_containers=1,
    scaledown_window=2,
)
def gpu_check_large(variant, release_id):
    return validate_on_gpu(variant, release_id)


@app.function(
    image=image, volumes={"/data": volume}, cpu=2, memory=4096, timeout=3600, max_containers=1, scaledown_window=2
)
def upload(release_id, repo_id):
    """only called when the owner explicitly runs --stage upload"""
    import os
    import sys
    from huggingface_hub import HfApi

    directory = release_path(release_id)
    sys.path.insert(0, "/opt/project/src")
    from clewen.release_utils import sha256_file, validate_release

    validate_release(directory, verify_checksums=True)
    report_path = Path("/data/release_checks") / release_id / "gpu_check.json"
    report = json.loads(report_path.read_text())
    if not report.get("passed") or report.get("release_manifest_sha256") != sha256_file(
        directory / "release_manifest.json"
    ):
        raise ValueError("run gpu_check on this exact release before uploading")
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.repo_info(repo_id, repo_type="model")
    if not info.private:
        raise ValueError("Upload target must be an existing PRIVATE model repository")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        commit_message="Upload Clewen inference release",
        delete_patterns=["clewen.py", "infer.py", "smoke_test.py", "example_decision.json"],
        parent_commit=info.sha,
        ignore_patterns=[".cache/**", "__pycache__/**", "*.pyc"],
    )
    return {"repo_id": repo_id, "private": True, "commit_url": commit.commit_url}


@app.local_entrypoint()
def main(stage: str = "unit_tests", variant: str = "clewen-flash", release_id: str = "", repo_id: str = ""):
    specs = json.loads((ROOT / "tools/models.json").read_text())
    if variant not in specs:
        raise ValueError(f"variant must be one of {list(specs)}")
    spec = specs[variant]
    release_id = release_id or spec["release_id"]
    if stage == "prepare":
        result = prepare.with_options(env={"HF_HUB_OFFLINE": "0"}).remote(variant)
    elif stage == "build":
        result = build.remote(variant, release_id)
    elif stage == "unit_tests":
        result = unit_tests.remote()
    elif stage == "recover":
        result = recover.remote(variant)
    elif stage == "gpu_check":
        checker = gpu_check_large if variant == "clewen" else gpu_check
        result = checker.remote(variant, release_id)
    elif stage == "upload":
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            raise ValueError("first log in using hf auth login or set HF_TOKEN")
        secret = modal.Secret.from_dict({"HF_TOKEN": token})
        result = upload.with_options(env={"HF_HUB_OFFLINE": "0"}, secrets=[secret]).remote(
            release_id, repo_id or spec["repo_id"]
        )
    else:
        raise ValueError("stage must be prepare, build, unit_tests, recover, gpu_check, or upload")
    output = ROOT / ".local/checks" / release_id
    output.mkdir(parents=True, exist_ok=True)
    (output / (stage + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"records", "text_before", "text_after", "decision"}},
            ensure_ascii=False,
            indent=2,
        )
    )

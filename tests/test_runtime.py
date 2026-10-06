"""small CPU checks for math, switching/error paths, and checkpoint export"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from clewen import modeling_clewen as clewen
from clewen.configuration_clewen import ClewenConfig
from clewen.adapter_runtime import install_adapters
from build_release import MODEL_SPECS, SOURCE_MODELS, build_release
from clewen.release_utils import copy_release, release_file, sha256_file, validate_release

PROJECT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def toy(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.LayerNorm(2)).eval()
        tensors = {
            "first/A": torch.randn(1, 3),
            "first/B": torch.randn(2, 1),
            "second/delta": torch.randn(2, 2),
            "norm/target": torch.tensor([1.4, 0.7]),
        }
        manifest = {
            "entries": [
                {"tensor": "0.weight", "file": "first", "kind": "low_rank", "shape": [2, 3]},
                {"tensor": "1.weight", "file": "second", "kind": "dense_delta", "shape": [2, 2]},
                {"tensor": "2.weight", "file": "norm", "kind": "parameter_override", "shape": [2]},
            ]
        }
        return model, tensors, manifest

    def test_adapter_math_and_exact_restoration(self):
        model, t, manifest = self.toy()
        x = torch.randn(7, 3)
        before = model(x).detach()
        originals = {name: parameter for name, parameter in model.named_parameters()}
        values = {name: parameter.detach().clone() for name, parameter in originals.items()}
        expected = F.linear(x, model[0].weight, model[0].bias) + F.linear(F.linear(x, t["first/A"]), t["first/B"])
        expected = F.linear(expected, model[1].weight, model[1].bias) + F.linear(expected, t["second/delta"])
        expected = F.layer_norm(expected, (2,), t["norm/target"], model[2].bias, model[2].eps)
        controller = install_adapters(model, self.root, manifest, t)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x), before))
            controller.set_enabled(True)
            self.assertTrue(torch.allclose(model(x), expected, atol=1e-6, rtol=1e-6))
            controller.set_enabled(False)
            self.assertTrue(torch.equal(model(x), before))
        for name, original in originals.items():
            self.assertIs(model.get_parameter(name), original)
            self.assertTrue(torch.equal(original, values[name]))

    def test_invalid_pack_does_not_partially_install(self):
        model, t, manifest = self.toy()
        t["second/delta"] = torch.zeros(4, 4)
        with self.assertRaises(ValueError):
            install_adapters(model, self.root, manifest, t)
        self.assertIsInstance(model[0], torch.nn.Linear)

    def test_device_and_dtype_moves_include_norm_overrides(self):
        model, tensors, manifest = self.toy()
        controller = install_adapters(model, self.root, manifest, tensors)
        model.add_module("controller", controller)
        model.to(dtype=torch.float64)
        controller.set_enabled(True)
        self.assertEqual(model[2].weight.dtype, torch.float64)
        controller.set_enabled(False)
        self.assertEqual(model[2].weight.dtype, torch.float64)

    def wrapper(self):
        base = torch.nn.Linear(2, 2)
        base.generation_config = GenerationConfig(eos_token_id=1, pad_token_id=0)
        controller = SimpleNamespace(enabled=False)
        controller.set_enabled = lambda value: setattr(controller, "enabled", value)
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1))
        return clewen.ClewenModel(
            ClewenConfig(),
            backbone=base,
            processor=processor,
            head=torch.nn.Identity(),
            controller=controller,
            directory=self.root,
        )

    def test_restore_on_forward_exception(self):
        wrapper = self.wrapper()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with wrapper._mode(True):
                self.assertTrue(wrapper._controller.enabled)
                raise RuntimeError("forward failed")
        self.assertFalse(wrapper._controller.enabled)

    def test_concurrent_modes_are_serialized(self):
        wrapper = self.wrapper()
        attempted, acquired = threading.Event(), threading.Event()
        errors = []

        def worker():
            try:
                attempted.set()
                with wrapper._mode(False):
                    self.assertFalse(wrapper._controller.enabled)
                    acquired.set()
            except BaseException as error:
                errors.append(error)

        with wrapper._mode(True):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(acquired.wait(0.05))
            self.assertTrue(wrapper._controller.enabled)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set())
        self.assertEqual(errors, [])

    def test_overlong_state_is_rejected_without_forward(self):
        wrapper = self.wrapper()
        record = {"state": "Important evidence at the end", "questions": {"ok": {"type": "noul"}}}
        with patch.object(clewen, "encode_record", return_value=SimpleNamespace(input_ids=tuple(range(50)))) as encoder:
            with self.assertRaisesRegex(ValueError, "not truncated"):
                wrapper.decision(record, max_input_tokens=10)
            self.assertEqual(encoder.call_args.kwargs["max_length"], 2**31 - 1)
        self.assertFalse(wrapper._controller.enabled)

    def test_all_decision_types_and_probability_mapping(self):
        wrapper = self.wrapper()
        record = {
            "state": {"amount": 5},
            "questions": {
                "yes": {"type": "noul"},
                "team": {"type": "choice", "criteria": {"z": "last", "a": "first"}},
                "level": {"type": "score", "criteria": ["low", "medium", "high"]},
            },
        }
        encoded = SimpleNamespace(
            input_ids=(1, 2, 3),
            questions=(
                SimpleNamespace(question_id="yes", option_ids=("true", "false")),
                SimpleNamespace(question_id="team", option_ids=("a", "z")),
                SimpleNamespace(question_id="level", option_ids=("0", "1", "2")),
            ),
        )

        class Decision(torch.nn.Module):
            def forward(self, batch):
                return [[torch.tensor([4.0, 0.0]), torch.tensor([0.0, 4.0]), torch.tensor([0.0, 0.0, 4.0])]]

        wrapper._decision_model = Decision()
        with (
            patch.object(clewen, "encode_record", return_value=encoded),
            patch.object(clewen, "collate_records", return_value={}),
        ):
            output = wrapper.decision(record)
        answers = output["answers"]
        self.assertIs(answers["yes"]["value"], True)
        self.assertEqual(answers["team"]["value"], "z")
        self.assertEqual(answers["level"]["value"], 2)
        self.assertGreater(answers["level"]["expected_score"], 1.9)
        self.assertEqual(output["usage"]["output_tokens"], 0)
        for answer in answers.values():
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=6)

    def source_fixture(self):
        qwen, clef, adapters = [self.root / name for name in ("qwen", "clef", "adapters")]
        for path in (qwen, clef, adapters):
            path.mkdir()
        (qwen / "config.json").write_text("{}")
        (qwen / "tokenizer_config.json").write_text("{}")
        (qwen / "tokenizer.json").write_text("{}")
        save_file({"mtp.x": torch.ones(2), "model.x": torch.zeros(2)}, str(qwen / "model.safetensors"))
        (qwen / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"mtp.x": "model.safetensors", "model.x": "model.safetensors"}})
        )
        (clef / "joint_head_config.json").write_text("{}")
        save_file(torch.nn.Linear(2, 2).state_dict(), str(clef / "joint_head.safetensors"))
        save_file({"target": torch.ones(2)}, str(adapters / "norm.safetensors"))
        manifest = {
            "models": SOURCE_MODELS,
            "artifact_id": "test",
            "entries": [
                {"tensor": "norm.weight", "kind": "parameter_override", "file": "norm.safetensors", "shape": [2]}
            ],
        }
        (adapters / "manifest.json").write_text(json.dumps(manifest))
        return qwen, clef, adapters

    def test_export_copy_preserves_shards_and_rejects_corruption(self):
        qwen, clef, adapters = self.source_fixture()
        output = self.root / "release"
        result = build_release(qwen, clef, adapters, PROJECT, output)
        self.assertEqual(result["mtp_tensor_count"], 1)
        validate_release(output, verify_checksums=True)
        copied = self.root / "saved"
        copy_release(output, copied)
        config = AutoConfig.from_pretrained(str(copied), trust_remote_code=True, local_files_only=True)
        self.assertEqual(config.auto_map["AutoModel"], "modeling_clewen.ClewenModel")
        self.assertEqual((copied / "qwen/model.safetensors").read_bytes(), (qwen / "model.safetensors").read_bytes())
        with self.assertRaises(FileExistsError):
            copy_release(output, copied)
        path = copied / "adapters/packed.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            validate_release(copied, verify_checksums=True)

    def test_export_rejects_wrong_source_revisions(self):
        qwen, clef, adapters = self.source_fixture()
        manifest = json.loads((adapters / "manifest.json").read_text())
        manifest["models"]["qwen"]["revision"] = "wrong"
        (adapters / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "pinned"):
            build_release(qwen, clef, adapters, PROJECT, self.root / "release")

    def test_large_variant_and_cached_release_identity(self):
        qwen, clef, adapters = self.source_fixture()
        manifest = json.loads((adapters / "manifest.json").read_text())
        manifest["models"] = MODEL_SPECS["clewen"]["sources"]
        (adapters / "manifest.json").write_text(json.dumps(manifest))
        output = self.root / "large"
        build_release(qwen, clef, adapters, PROJECT, output, variant="clewen")
        config, _ = validate_release(output)
        self.assertEqual(config["name"], "Clewen")
        self.assertEqual(config["auto_map"]["AutoModelForCausalLM"], "modeling_clewen.ClewenModel")
        manifest["artifact_id"] = "another"
        (adapters / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(FileExistsError, "another model"):
            build_release(qwen, clef, adapters, PROJECT, output, variant="clewen")

    def test_auto_model_repo_id_loads_cached_dynamic_code_and_one_backbone(self):
        qwen, clef, adapters = self.source_fixture()
        repo_id, commit = "test-org/clewen-flash", "a" * 40
        cache = self.root / "hf-cache"
        repo = cache / "models--test-org--clewen-flash"
        snapshot = repo / "snapshots" / commit
        build_release(qwen, clef, adapters, PROJECT, snapshot)
        self.link_snapshot(snapshot)
        self.assertTrue((snapshot / ".gitattributes").is_symlink())
        self.assertFalse((snapshot / ".gitattributes").resolve().is_relative_to(snapshot))
        (repo / "refs").mkdir()
        (repo / "refs/main").write_text(commit)
        model_class = get_class_from_dynamic_module(
            "modeling_clewen.ClewenModel", repo_id, cache_dir=str(cache), local_files_only=True
        )
        import sys

        dynamic = sys.modules[model_class.__module__]

        def backbone(*args, **kwargs):
            base = torch.nn.Module()
            base.norm = torch.nn.LayerNorm(2)
            base.generation_config = GenerationConfig(eos_token_id=1, pad_token_id=0)
            return base

        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1))
        with (
            patch.object(dynamic.Qwen3_5ForConditionalGeneration, "from_pretrained", side_effect=backbone) as load,
            patch.object(dynamic.AutoProcessor, "from_pretrained", return_value=processor),
            patch.object(dynamic, "JointSchemaHead", side_effect=lambda **kwargs: torch.nn.Linear(2, 2)),
        ):
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=str(cache),
                local_files_only=True,
                device_map="cpu",
                dtype=torch.float32,
            )
        self.assertEqual(load.call_count, 1)
        self.assertEqual(model.directory, snapshot)
        self.assertIs(model.backbone, model._decision_model.language_model)
        self.assertEqual(model.dtype, torch.float32)
        self.assertIn("transformers_modules", type(model).__module__)
        self.assertFalse(model._controller.enabled)
        saved = self.root / "saved-hf"
        model.save_pretrained(saved)
        config = AutoConfig.from_pretrained(saved, trust_remote_code=True, local_files_only=True)
        self.assertEqual(config.model_type, "clewen")

    def link_snapshot(self, snapshot):
        """real HF layout: snapshots/<commit>/<file> -> repo/blobs/<hash>"""
        blobs = snapshot.parent.parent / "blobs"
        blobs.mkdir(exist_ok=True)
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                blob = blobs / sha256_file(path)
                if blob.exists():
                    path.unlink()
                else:
                    path.rename(blob)
                path.symlink_to(os.path.relpath(blob, path.parent))

    def test_manifest_paths_still_reject_traversal_and_absolute_names(self):
        for name in (
            "",
            ".",
            "..",
            "../config.json",
            "qwen/../../outside",
            "/tmp/weights",
            "qwen/../config.json",
            "qwen\\..\\outside",
            "bad\0name",
            None,
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                release_file(self.root, name)

    def test_symlink_snapshot_keeps_integrity_checks(self):
        qwen, clef, adapters = self.source_fixture()
        snapshot = self.root / "cache/models--test--clewen/snapshots" / ("b" * 40)
        build_release(qwen, clef, adapters, PROJECT, snapshot)
        self.link_snapshot(snapshot)
        validate_release(snapshot, verify_checksums=True)
        original = (snapshot / "qwen/model.safetensors").read_bytes()
        copied = self.root / "saved-symlinks"
        copy_release(snapshot, copied)
        self.assertFalse((copied / "qwen/model.safetensors").is_symlink())
        self.assertEqual((copied / "qwen/model.safetensors").read_bytes(), original)
        # same-length corruption must still be caught through the symlink
        weight_blob = (snapshot / "adapters/packed.safetensors").resolve()
        data = bytearray(weight_blob.read_bytes())
        data[-1] ^= 1
        weight_blob.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            validate_release(snapshot, verify_checksums=True)
        # broken cache links must fail rather than being skipped
        weight_blob.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or incomplete"):
            validate_release(snapshot)

    def test_standard_generate_uses_text_and_restores_on_error(self):
        model = self.wrapper()

        def generate(*args, **kwargs):
            self.assertFalse(model._controller.enabled)
            raise RuntimeError("Generation failed")

        model.backbone.generate = generate
        with self.assertRaises(RuntimeError):
            model.generate(torch.tensor([[1]]))
        self.assertFalse(model._controller.enabled)

    def test_schema_validation_and_no_auto(self):
        wrapper = self.wrapper()
        with self.assertRaises(ValueError):
            wrapper.run(mode="auto")
        with self.assertRaises(ValueError):
            wrapper.decision({"state": "x", "questions": {"q": {"type": "score", "criteria": {"0": "low"}}}})


if __name__ == "__main__":
    unittest.main()

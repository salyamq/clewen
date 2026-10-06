"""clewen inference: one Qwen backbone, explicit text/decision modes"""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GenerationMixin, PreTrainedModel, Qwen3_5ForConditionalGeneration

from .configuration_clewen import ClewenConfig
from .joint_schema_model import ClefModel, JointSchemaHead, encode_record, collate_records, question_options
from .adapter_runtime import install_adapters
from .release_utils import copy_release, validate_release


def validate_questions(record):
    if not isinstance(record, dict) or "state" not in record:
        raise ValueError("decision requires a record with state and questions")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty dictionary")
    for name, question in questions.items():
        if not isinstance(name, str) or not name or not isinstance(question, dict):
            raise ValueError("Question IDs must be nonempty strings; questions must be dictionaries")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind not in {"noul", "choice", "score"}:
            raise ValueError(f"{name}: type must be noul, choice, or score")
        if kind == "choice" and (
            not isinstance(criteria, dict) or not criteria or any(not isinstance(k, str) or not k for k in criteria)
        ):
            raise ValueError(f"{name}: choice criteria must map nonempty string IDs to descriptions")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise ValueError(f"{name}: score criteria must be a nonempty ordered list")
        if (
            kind == "noul"
            and criteria is not None
            and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"})
        ):
            raise ValueError(f"{name}: noul criteria may only describe true and false")


class ClewenModel(PreTrainedModel, GenerationMixin):
    """inference-only wrapper. Requests on an instance execute sequentially

    Use run(mode='text', messages=...) or run(mode='decision', record=...)
    Does not choose modes, generate schemas, or apply confidence thresholds"""

    config_class = ClewenConfig
    base_model_prefix = "backbone"
    _supports_sdpa = True

    def __init__(self, config, backbone=None, processor=None, head=None, controller=None, directory=None):
        super().__init__(config)
        if backbone is None:
            raise ValueError("Load this inference release with AutoModel.from_pretrained(...)")
        self.backbone = backbone
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.head = head
        self._decision_model = ClefModel(backbone, head).eval()
        self._controller = controller
        self._lock = threading.RLock()
        self.directory = Path(directory)
        self.generation_config = backbone.generation_config

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path,
        *model_args,
        config=None,
        device=None,
        device_map=None,
        dtype=None,
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        attn_implementation="sdpa",
        verify_checksums=False,
        force_download=False,
        **kwargs,
    ):
        """load a complete Clewen repo or local export; never fetch two backbones"""
        if model_args:
            raise TypeError("This inference release does not accept positional model arguments")
        # autoModel supplies these internal options to a custom model loader
        commit = kwargs.pop("_commit_hash", None) or getattr(config, "_commit_hash", None)
        for key in ("trust_remote_code", "_from_auto", "_from_pipeline", "low_cpu_mem_usage", "use_safetensors"):
            kwargs.pop(key, None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)
        subfolder = kwargs.pop("subfolder", "")
        legacy_dtype = kwargs.pop("torch_dtype", None)
        if adapter_kwargs or subfolder:
            raise ValueError("Load the complete Clewen repo root; external PEFT adapters are unsupported")
        if kwargs:
            raise TypeError(f"Unsupported load options: {', '.join(sorted(kwargs))}")
        if device_map is not None:
            if isinstance(device_map, dict) and set(device_map) == {""}:
                mapped_device = device_map[""]
            elif device_map == "auto":
                mapped_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif isinstance(device_map, (str, torch.device)):
                mapped_device = device_map
            else:
                raise ValueError(
                    "Use device_map='cuda', 'cuda:0', 'cpu', or {'': device}; multi-device dispatch is unsupported"
                )
            mapped_device = f"cuda:{mapped_device}" if isinstance(mapped_device, int) else mapped_device
            if device is not None and torch.device(device) != torch.device(mapped_device):
                raise ValueError("device and device_map disagree")
            device = mapped_device
        device = torch.device("cuda" if device is None else device)
        directory = Path(model_id_or_path).expanduser()
        if not directory.is_dir():
            # pin weights to the SAME commit whose config/code AutoModel resolved
            directory = Path(
                snapshot_download(
                    str(model_id_or_path),
                    revision=commit or revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            )
        saved_config, _ = validate_release(directory, verify_checksums=verify_checksums)
        config = ClewenConfig.from_dict(saved_config) if config is None else config
        dtype = dtype or legacy_dtype or getattr(config, "dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, dtype)
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("Clewen supports a single CUDA GPU or CPU")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA build of PyTorch")
        if dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("Use BF16 on a GPU or float32 for CPU debugging")
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            directory / "qwen",
            dtype=dtype,
            device_map={"": str(device)},
            attn_implementation=attn_implementation,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(directory / "qwen", local_files_only=True)
        head = JointSchemaHead(**json.loads((directory / "head/joint_head_config.json").read_text()))
        head.load_state_dict(load_file(str(directory / "head/joint_head.safetensors")), strict=True)
        head = head.to(device=device, dtype=dtype).eval()
        manifest = json.loads((directory / "adapters/manifest.json").read_text())
        packed = load_file(str(directory / "adapters/packed.safetensors"))
        controller = install_adapters(backbone, directory / "adapters", manifest, packed_tensors=packed)
        controller.set_enabled(False)
        backbone.eval().requires_grad_(False)
        head.requires_grad_(False)
        model = cls(
            config, backbone=backbone, processor=processor, head=head, controller=controller, directory=directory
        )
        return model.eval()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def forward(self, *args, mode="text", record=None, **kwargs):
        if mode == "decision":
            if args or record is None:
                raise ValueError("decision forward requires record=...")
            return self.decision(record, **kwargs)
        if mode != "text" or record is not None:
            raise ValueError("Tensor forward uses text mode; use record with decision mode")
        with self._mode(False):
            return self.backbone(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """standard HF tensor generation through the unchanged Qwen branch"""
        with self._mode(False):
            return self.backbone.generate(*args, **kwargs)

    @contextmanager
    def _mode(self, decision):
        # hold the lock for the whole forward/generation, including exception cleanup
        with self._lock:
            try:
                self._controller.set_enabled(decision)
                with torch.inference_mode():
                    yield
            finally:
                self._controller.set_enabled(False)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, *, mode, **kwargs):
        if mode == "text":
            return self.text(**kwargs)
        if mode == "decision":
            return self.decision(**kwargs)
        raise ValueError("mode must be 'text' or 'decision'")

    def prepare_chat(self, messages, *, thinking=False, max_input_tokens=None, **processor_kwargs):
        """encode chat messages with the original Qwen tokenizer and media processor"""
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} or "content" not in m
                for m in messages
            )
        ):
            raise ValueError("messages must be a nonempty chat message list")
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        if {"past_key_values", "input_ids", "inputs_embeds", "return_tensors", "tokenize"} & processor_kwargs.keys():
            raise ValueError("External KV caches and pretokenized inputs are unsupported")
        limit = self.config.default_max_input_tokens if max_input_tokens is None else max_input_tokens
        if type(limit) is not int or limit < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if all(isinstance(m["content"], str) for m in messages):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
                return_dict=True,
                return_tensors="pt",
                **processor_kwargs,
            )
        inputs = inputs.to(self.device)
        input_count = inputs["input_ids"].shape[-1]
        if input_count > limit:
            raise ValueError(f"Input has {input_count} tokens; limit is {limit}. Input was not truncated.")
        return inputs

    def text(
        self,
        messages,
        *,
        thinking=False,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        max_input_tokens=None,
        **processor_kwargs,
    ):
        """generate with unchanged Qwen. Multimodal messages use the Qwen processor"""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(do_sample) is not bool:
            raise ValueError("do_sample must be a boolean")
        with self._mode(False):
            self._sync()
            start = time.perf_counter()
            inputs = self.prepare_chat(
                messages, thinking=thinking, max_input_tokens=max_input_tokens, **processor_kwargs
            )
            input_count = inputs["input_ids"].shape[-1]
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            generation = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": pad,
            }
            if do_sample:
                generation.update(temperature=temperature, top_p=top_p)
            output = self.backbone.generate(**inputs, **generation)
            ids = output[0, input_count:].tolist()
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            reasoning, answer = "", decoded
            if "</think>" in decoded:
                reasoning, answer = decoded.split("</think>", 1)
                reasoning = reasoning.removeprefix("<think>").strip()
            elif thinking:
                reasoning, answer = decoded.removeprefix("<think>").strip(), ""
            eos = self.backbone.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            self._sync()
            return {
                "mode": "text",
                "answer": answer.strip(),
                "thinking": reasoning,
                "raw_text": decoded,
                "token_ids": ids,
                "truncated": not ids or ids[-1] not in eos,
                "usage": {"input_tokens": input_count, "output_tokens": len(ids)},
                "seconds": time.perf_counter() - start,
            }

    def decision(self, record, *, max_input_tokens=None):
        """return every field's logits/probabilities in one joint decision pass

        state may be text or JSON. Optional images/videos follow official Clef format
        Oversized input raises instead of silently dropping the end of the state"""
        validate_questions(record)
        limit = self.config.default_max_input_tokens if max_input_tokens is None else max_input_tokens
        if type(limit) is not int or limit < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        with self._mode(True):
            self._sync()
            start = time.perf_counter()
            # the upstream encoder truncates state. Encode with a generous bound first,
            # then reject over-limit records so important evidence is never dropped
            encoded = encode_record(self.tokenizer, record, max_length=2**31 - 1, processor=self.processor)
            count = len(encoded.input_ids)
            if count > limit:
                raise ValueError(f"Decision input has {count} tokens; limit is {limit}. Input was not truncated.")
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            batch = collate_records([encoded], pad, self.device)
            logits = self._decision_model(batch)[0]
            answers = {}
            for question, values in zip(encoded.questions, logits):
                values = values.float()
                probabilities = dict(zip(question.option_ids, values.softmax(-1).tolist()))
                selected = max(probabilities, key=probabilities.__getitem__)
                schema = record["questions"][question.question_id]
                result = {
                    "type": schema["type"],
                    "selected": selected,
                    "confidence": probabilities[selected],
                    "probabilities": probabilities,
                    "logits": dict(zip(question.option_ids, values.tolist())),
                }
                if schema["type"] == "noul":
                    result["value"] = selected == "true"
                elif schema["type"] == "score":
                    result["value"] = int(selected)
                    result["expected_score"] = sum(int(k) * p for k, p in probabilities.items())
                    result["legend"] = dict(question_options(schema))
                else:
                    result["value"] = selected
                answers[question.question_id] = result
            self._sync()
            return {
                "mode": "decision",
                "answers": answers,
                "usage": {"input_tokens": count, "output_tokens": 0},
                "seconds": time.perf_counter() - start,
            }

    def save_pretrained(self, directory, **kwargs):
        """save this inference release by copying its original immutable files

        Do not serialize backbone.state_dict(): installed adapters wrap its modules
        MTP is preserved on disk even if the Transformers loader ignores it"""
        if kwargs:
            raise TypeError(
                "This inference-only save copies the complete original release; save options are unsupported"
            )
        with self._lock:
            return copy_release(self.directory, directory)

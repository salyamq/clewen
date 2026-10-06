"""check published HF interfaces against the original Clef checkpoint"""

import gc
import json
import statistics
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen
import io

import torch
from fastapi.testclient import TestClient
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from clewen.joint_schema_model import ClefModel, JointSchemaHead, collate_records, encode_record
from clewen.server import create_app
from safetensors.torch import load_file
from smoke_check import check as smoke_check


def original_decision(model, processor, record):
    encoded = encode_record(processor.tokenizer, record, processor=processor)
    batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
    with torch.inference_mode():
        logits = model(batch)[0]
    return {
        question.question_id: dict(zip(question.option_ids, values.float().softmax(-1).tolist()))
        for question, values in zip(encoded.questions, logits)
    }


@contextmanager
def plain_backbone(model):
    restored = []
    for name, module in list(model.named_modules()):
        if hasattr(module, "base") and hasattr(module, "enabled"):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            restored.append((parent, child, module))
            setattr(parent, child, module.base)
    try:
        yield
    finally:
        for parent, child, module in restored:
            setattr(parent, child, module)


def check(directory, clef_directory, variant, photos):
    start = time.monotonic()
    fixture = json.loads((Path(__file__).parent / "fixtures/decision_cases.json").read_text())
    records = [{"state": case["state"], "questions": case["questions"]} for case in fixture]
    vision = []
    photos.mkdir(parents=True, exist_ok=True)
    sources = {
        "cats": "https://raw.githubusercontent.com/huggingface/transformers/v5.10.2/tests/fixtures/tests_samples/COCO/000000039769.png",
        "astronaut": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png",
    }
    for name, expected in [
        ("cats", {"cat": "true", "person": "false"}),
        ("astronaut", {"cat": "false", "person": "true"}),
    ]:
        image_path = photos / (name + ".png")
        if not image_path.is_file():
            with urlopen(sources[name], timeout=30) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
            image.thumbnail((448, 448))
            image.save(image_path)
        image = Image.open(image_path).convert("RGB")
        record = {
            "state": "Use the attached photograph.",
            "images": [image],
            "questions": {
                "cat": {"type": "noul", "instructions": "Is a cat visible?"},
                "person": {"type": "noul", "instructions": "Is a person visible?"},
            },
        }
        records.append(record)
        vision.append((name, image, expected))

    print("loading original clef for comparison", flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        clef_directory, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    processor = AutoProcessor.from_pretrained(clef_directory, local_files_only=True)
    head = JointSchemaHead(**json.loads((clef_directory / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(str(clef_directory / "joint_head.safetensors")), strict=True)
    original = ClefModel(base, head.to("cuda", torch.bfloat16)).eval()
    reference = [original_decision(original, processor, record) for record in records]
    del original, base, head, processor
    gc.collect()
    torch.cuda.empty_cache()

    print("loading clewen through AutoModelForCausalLM", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        directory, trust_remote_code=True, device_map="auto", dtype="bfloat16", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    assert tokenizer.encode("hello") == model.tokenizer.encode("hello")
    assert type(processor) is type(model.processor)
    result = smoke_check(model)
    comparisons, matches, questions, max_difference = [], 0, 0, 0.0
    original_correct, recovered_correct, labeled = 0, 0, 0
    for index, (record, baseline) in enumerate(zip(records, reference)):
        output = model.decision(record)
        comparisons.append(
            {
                "id": fixture[index]["id"] if index < len(fixture) else vision[index - len(fixture)][0],
                "original": baseline,
                "recovered": output["answers"],
            }
        )
        expected = fixture[index]["reference"] if index < len(fixture) else vision[index - len(fixture)][2]
        for name, probabilities in baseline.items():
            answer = output["answers"][name]
            selected = max(probabilities, key=probabilities.get)
            matches += answer["selected"] == selected
            questions += 1
            for option, probability in probabilities.items():
                max_difference = max(max_difference, abs(probability - answer["probabilities"][option]))
        for name, label in expected.items():
            labeled += 1
            original_correct += max(baseline[name], key=baseline[name].get) == label
            recovered_correct += output["answers"][name]["selected"] == label

    assert matches / questions >= 0.98, (matches, questions)
    assert max_difference < 0.03, max_difference
    assert recovered_correct >= original_correct - 1, (original_correct, recovered_correct)

    messages = [{"role": "user", "content": "Reply with the number 42."}]
    inputs = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    actual = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    with plain_backbone(model.backbone), torch.inference_mode():
        expected = model.backbone.generate(**inputs, max_new_tokens=16, do_sample=False)
    assert torch.equal(actual, expected), "text differs from unwrapped Qwen"

    captions = []
    for name, image, _ in vision:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the main subject in one short sentence."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        caption = processor.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        assert (
            "cat" in caption.lower()
            if name == "cats"
            else any(word in caption.lower() for word in ["astronaut", "person", "woman"])
        )
        captions.append({"id": name, "text": caption})

    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module

    assert qwen_module.is_fast_path_available
    fast_blocks = [
        module for module in model.backbone.modules() if isinstance(module, qwen_module.Qwen3_5GatedDeltaNet)
    ]
    assert all(module.chunk_gated_delta_rule is qwen_module.chunk_gated_delta_rule for module in fast_blocks)
    for _ in range(2):
        model.decision(records[0])
    times = [model.decision(records[0])["seconds"] for _ in range(5)]

    client = TestClient(create_app(model, variant))
    body = {"model": variant, "messages": [{"role": "user", "content": "Reply with the number 42."}], "max_tokens": 16}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert "42" in response.json()["choices"][0]["message"]["content"]
    stream = client.post(
        "/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}
    )
    assert stream.status_code == 200 and "[DONE]" in stream.text and '"completion_tokens"' in stream.text, stream.text
    decision = client.post("/v1/decisions", json={"model": variant, **records[0]})
    assert decision.status_code == 200, decision.text
    assert decision.json()["answers"] == model.decision(records[0])["answers"]

    result.update(
        {
            "passed": True,
            "variant": variant,
            "decision_records": len(records),
            "agreement": {"matching_questions": matches, "questions": questions},
            "labels": {"original_correct": original_correct, "clewen_correct": recovered_correct, "total": labeled},
            "max_probability_difference": max_difference,
            "records": comparisons,
            "root_auto_tokenizer": True,
            "root_auto_processor": True,
            "auto_model_for_causal_lm": True,
            "text_matches_plain_qwen": True,
            "photo_captions": captions,
            "http_chat_decision_stream": True,
            "fast_linear_blocks": len(fast_blocks),
            "decision_median_seconds": statistics.median(times),
            "decision_timing_samples": times,
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "packages": {
                name: version(name) for name in ["torch", "transformers", "flash-linear-attention", "causal-conv1d"]
            },
        }
    )
    return result

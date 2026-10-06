"""gPU integration assertions, kept outside the user-facing HF release"""

import json
from pathlib import Path


def check(model):
    messages = [{"role": "user", "content": "Return only the result of 17 + 25."}]
    before = model.run(mode="text", messages=messages, max_new_tokens=32)
    case = json.loads((Path(__file__).parent / "fixtures/decision_cases.json").read_text())[1]
    record = {"state": case["state"], "questions": case["questions"]}
    decision = model.run(mode="decision", record=record)
    after = model.run(mode="text", messages=messages, max_new_tokens=32)
    assert before["token_ids"] == after["token_ids"], "decision -> text changed Qwen output"
    assert decision["answers"]["department"]["selected"] == "billing"
    assert decision["answers"]["urgent"]["selected"] == "false"
    for answer in decision["answers"].values():
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-5
    try:
        model.decision(record, max_input_tokens=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Oversized decision was not rejected")
    assert not model._controller.enabled
    assert before["token_ids"] == model.text(messages, max_new_tokens=32)["token_ids"]
    assert model._decision_model.language_model is model.backbone
    # standard hf generate accepts tensors too  without our chat convenience api
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, do_sample=False, max_new_tokens=32, use_cache=True, pad_token_id=model.tokenizer.pad_token_id
    )
    assert output[0, inputs["input_ids"].shape[-1] :].tolist() == before["token_ids"]
    return {
        "text_before": before,
        "decision": decision,
        "text_after": after,
        "one_backbone": True,
        "text_tokens_identical": True,
        "exception_restores_text": True,
        "standard_generate_matches": True,
    }

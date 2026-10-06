"""standard Hugging Face text and image generation"""

import argparse

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai-slab/clewen-flash")
    parser.add_argument("--prompt", default="Explain what a gradient is.")
    parser.add_argument("--image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device, dtype="bfloat16"
    )
    if args.image:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        content = [{"type": "image", "path": args.image}, {"type": "text", "text": args.prompt}]
    else:
        processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        content = args.prompt
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        enable_thinking=args.thinking,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()

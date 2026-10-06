"""typed decisions from text, JSON, and optional images"""

import argparse
import json
from pathlib import Path

from PIL import Image
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai-slab/clewen-flash")
    parser.add_argument("--record", type=Path, default=Path(__file__).with_name("record.json"))
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    if args.image:
        record["images"] = [Image.open(path).convert("RGB") for path in args.image]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device, dtype="bfloat16"
    )
    print(json.dumps(model.decision(record), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

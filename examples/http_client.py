"""openAI client for text and an HTTP request for decisions"""

import argparse
import json
from pathlib import Path

import httpx
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai-slab/clewen-flash")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--mode", choices=["text", "decision"], default="text")
    args = parser.parse_args()
    if args.mode == "text":
        client = OpenAI(base_url=args.url + "/v1", api_key=args.api_key)
        stream = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": "What is a gradient?"}], max_tokens=256, stream=True
        )
        for chunk in stream:
            if chunk.choices:
                print(chunk.choices[0].delta.content or "", end="", flush=True)
        print()
    else:
        record = json.loads(Path(__file__).with_name("record.json").read_text())
        response = httpx.post(
            args.url + "/v1/decisions",
            json={"model": args.model, **record},
            timeout=120,
            headers={"Authorization": "Bearer " + args.api_key},
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()

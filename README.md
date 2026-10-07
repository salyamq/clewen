The decision mode uses recovered switchable adapters and the original Clef joint schema head. No additional training was performed. Recovered adapters approximate the learned changes; they are not the original LoRA factors.

## Usage

Both modes support text and images. Decision inputs can also contain JSON. Modes are selected explicitly:

- **Qwen:** text generation and image understanding.
- **Clef:** structured answers with probabilities for `choice`, `noul` (yes/no), and `score` questions.

Install on Linux with Python 3.11 or 3.12 and a CUDA GPU:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install "clewen[cuda] @ git+https://github.com/salyamq/clewer.git"
```

```python
from PIL import Image
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "ai-slab/clewen-flash",  # or "ai-slab/clewen"
    trust_remote_code=True,
    device_map="cuda",
    dtype="bfloat16",
)

# Text generation
reply = model.text(
    [{"role": "user", "content": "Explain gradient descent briefly."}],
    max_new_tokens=128,
)
print(reply["answer"])

# Image understanding
image = Image.open("example.jpg").convert("RGB")
reply = model.text(
    [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }],
    max_new_tokens=128,
)
print(reply["answer"])

# Structured decision from an image
result = model.decision({
    "state": "Inspect the attached image.",
    "images": [image],
    "questions": {
        "cat_visible": {
            "type": "noul",
            "instructions": "Is a cat visible?",
        },
    },
})
print(result["answers"])
```

For text or JSON decisions, omit `images` and put your information in `state`. Both modes use the same loaded model instance.

## Results

Decision-mode scores from complete evaluation runs. Scores are percentages; higher is better. Workflow exact-action scores require the complete action set to match the reference labels.

| Benchmark / metric | Clewen | Clewen-Flash |
| --- | ---: | ---: |
| BFCL — case exact accuracy | 98.41 | 98.88 |
| API-Bank — accuracy | 91.73 | 93.11 |
| BANKING77 — macro-F1 | 94.08 | 90.80 |
| RAGTruth — hallucination F1 | 79.30 | 35.74 |
| When2Call — accuracy | 72.48 | 65.36 |
| Invoice processing — exact actions | 64.67 | 57.11 |
| Invoice processing — primary action | 86.44 | 74.22 |
| Customer service — exact actions | 76.31 | 76.96 |
| Security incidents — exact actions | 63.33 | 61.67 |
| Agent trace observability — primary action | 68.47 | 69.82 |
| Decision latency — median, ms ↓ | 142.30 | 78.06 |
| Decision latency — p95, ms ↓ | 274.15 | 111.26 |

Latency was measured on one H200 in BF16 across the five Decision Index benchmarks. It includes input encoding and answer decoding, excluding model loading and warmup. These evaluations measure structured decisions, not text-generation quality.

## License

Apache-2.0. Qwen components are credited to the Qwen authors; Clef components are credited to Cloudflare. See `LICENSE`, `LICENSE-QWEN`, `LICENSE-CLEF`, and `NOTICE` for attribution and license details.

Clewen is an independent project, not an official Qwen or Cloudflare release.

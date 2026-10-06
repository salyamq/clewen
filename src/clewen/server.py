"""hTTP inference for one Clewen instance"""

import argparse
import asyncio
import base64
import io
import json
import secrets
import threading
import time
import uuid
from queue import Empty

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[dict]
    max_tokens: int = Field(default=512, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False
    stream_options: dict | None = None
    thinking: bool = False
    n: int = Field(default=1, ge=1, le=1)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: object
    questions: dict
    images: list[str] = Field(default_factory=list)
    media_kwargs: dict = Field(default_factory=dict)
    max_input_tokens: int | None = Field(default=None, ge=1)


class Cancelled(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def decode_image(value):
    if not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("decision images must be base64 image data URLs")
    data = base64.b64decode(value.split(";base64,", 1)[1], validate=True)
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except OSError as error:
        raise ValueError("decision image data could not be decoded") from error


def create_app(model, model_id, api_key=None):
    app = FastAPI(title="Clewen", version="0.3.0")

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if api_key and not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {api_key}"):
            return JSONResponse({"error": {"message": "invalid API key", "type": "authentication_error"}}, 401)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_input(request, error):
        return JSONResponse({"error": {"message": str(error), "type": "invalid_request_error"}}, 400)

    def check_model(requested):
        if requested != model_id:
            raise HTTPException(404, detail=f"model must be {model_id}")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/decisions")
    async def decisions(body: DecisionRequest):
        check_model(body.model)
        record = {"state": body.state, "questions": body.questions, "media_kwargs": body.media_kwargs}
        record["images"] = await run_in_threadpool(lambda: [decode_image(value) for value in body.images])
        result = await run_in_threadpool(model.decision, record, max_input_tokens=body.max_input_tokens)
        return {"model": model_id, **result}

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        check_model(body.model)
        maximum = body.max_completion_tokens or body.max_tokens
        common = {
            "thinking": body.thinking,
            "max_new_tokens": maximum,
            "do_sample": body.temperature > 0,
            "temperature": body.temperature,
            "top_p": body.top_p,
        }
        response_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

        def envelope():
            return {"id": response_id, "object": "chat.completion", "created": created, "model": model_id}

        def usage(result):
            counts = result["usage"]
            return {
                "prompt_tokens": counts["input_tokens"],
                "completion_tokens": counts["output_tokens"],
                "total_tokens": counts["input_tokens"] + counts["output_tokens"],
            }

        if not body.stream:
            result = await run_in_threadpool(model.text, body.messages, **common)
            message = {"role": "assistant", "content": result["answer"]}
            if result["thinking"]:
                message["reasoning_content"] = result["thinking"]
            return {
                **envelope(),
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "length" if result["truncated"] else "stop"}
                ],
                "usage": usage(result),
            }

        inputs = await run_in_threadpool(model.prepare_chat, body.messages, thinking=body.thinking)
        streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.25)
        cancelled = threading.Event()
        outcome = {}
        generation = {
            "max_new_tokens": maximum,
            "do_sample": common["do_sample"],
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([Cancelled(cancelled)]),
            "use_cache": True,
        }
        if common["do_sample"]:
            generation.update(temperature=body.temperature, top_p=body.top_p)

        def generate():
            try:
                output = model.generate(**inputs, **generation)
                outcome["tokens"] = output[0, inputs["input_ids"].shape[-1] :].tolist()
            except Exception as error:
                outcome["error"] = str(error)
                streamer.end()

        def event(delta=None, finish=None, counts=None):
            data = {
                **envelope(),
                "object": "chat.completion.chunk",
                "choices": [] if counts else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            }
            if counts:
                data["usage"] = counts
            return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

        async def stream():
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            try:
                yield event({"role": "assistant", "content": ""})
                while True:
                    if await request.is_disconnected():
                        cancelled.set()
                        break
                    try:
                        item = await run_in_threadpool(streamer.text_queue.get, True, 0.25)
                    except Empty:
                        continue
                    if item == streamer.stop_signal:
                        break
                    if item:
                        yield event({"content": item})
                await run_in_threadpool(thread.join)
                if "error" in outcome:
                    yield (
                        "data: " + json.dumps({"error": {"message": outcome["error"], "type": "server_error"}}) + "\n\n"
                    )
                else:
                    tokens = outcome.get("tokens", [])
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    yield event(finish="stop" if tokens and tokens[-1] in eos else "length")
                    if body.stream_options and body.stream_options.get("include_usage"):
                        counts = {
                            "usage": {"input_tokens": inputs["input_ids"].shape[-1], "output_tokens": len(tokens)}
                        }
                        yield event(counts=usage(counts))
                yield "data: [DONE]\n\n"
            finally:
                cancelled.set()
                await asyncio.shield(run_in_threadpool(thread.join))

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    import os

    import uvicorn
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Serve text chat and Clef decisions from one model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, revision=args.revision, device_map=args.device, dtype=args.dtype
    )
    uvicorn.run(create_app(model, args.model, api_key=os.environ.get("CLEWEN_API_KEY")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""hTTP protocol and streaming checks without model weights"""

import json
import unittest
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from clewen.server import create_app


class ServerTests(unittest.TestCase):
    def setUp(self):
        tokenizer = SimpleNamespace(
            decode=lambda ids, **kwargs: "".join({3: "hello ", 4: "world "}.get(i, "") for i in ids)
        )
        self.model = SimpleNamespace(tokenizer=tokenizer, generation_config=SimpleNamespace(eos_token_id=9))
        self.model.text = lambda messages, **kwargs: {
            "answer": "hello world",
            "thinking": "",
            "truncated": False,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
        self.model.prepare_chat = lambda messages, **kwargs: {"input_ids": torch.tensor([[1, 2]])}
        self.model.decision = lambda record, **kwargs: {"mode": "decision", "answers": {"ok": {"value": True}}}

        def generate(**kwargs):
            streamer = kwargs["streamer"]
            streamer.put(kwargs["input_ids"])
            for token in [3, 4, 9]:
                streamer.put(torch.tensor([token]))
            streamer.end()
            return torch.tensor([[1, 2, 3, 4, 9]])

        self.model.generate = generate
        self.client = TestClient(create_app(self.model, "test"))
        self.body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}

    def test_chat_protocol_and_usage(self):
        response = self.client.post("/v1/chat/completions", json=self.body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(data["usage"]["total_tokens"], 5)

    def test_stream_protocol_and_usage(self):
        response = self.client.post(
            "/v1/chat/completions", json={**self.body, "stream": True, "stream_options": {"include_usage": True}}
        )
        lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in lines[:-1]]
        content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"])
        self.assertEqual(content, "hello world ")
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)

    def test_stream_error_is_reported(self):
        def fail(**kwargs):
            raise RuntimeError("failed")

        self.model.generate = fail
        response = self.client.post("/v1/chat/completions", json={**self.body, "stream": True})
        self.assertIn('"message": "failed"', response.text)
        self.assertIn("[DONE]", response.text)

    def test_decision_and_invalid_media(self):
        body = {"model": "test", "state": "hi", "questions": {"ok": {"type": "noul"}}}
        self.assertTrue(self.client.post("/v1/decisions", json=body).json()["answers"]["ok"]["value"])
        response = self.client.post("/v1/decisions", json={**body, "images": ["not an image"]})
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/v1/decisions", json={**body, "images": ["data:image/png;base64,bm90YW5pbWFnZQ=="]}
        )
        self.assertEqual(response.status_code, 400)

    def test_auth_model_and_unsupported_options(self):
        self.assertEqual(
            self.client.post("/v1/chat/completions", json={**self.body, "model": "other"}).status_code, 404
        )
        self.assertEqual(self.client.post("/v1/chat/completions", json={**self.body, "tools": []}).status_code, 422)
        client = TestClient(create_app(self.model, "test", api_key="secret"))
        self.assertEqual(client.get("/v1/models").status_code, 401)
        self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code, 200)

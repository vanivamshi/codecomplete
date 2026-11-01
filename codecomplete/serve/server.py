"""Completion server for the VS Code extension.

    python -m codecomplete.serve.server --model artifacts/merged --port 8000

POST /complete   {"prefix", "suffix", "mode": "line"|"block"} -> {"id", "text", "latency_ms"}
POST /telemetry  {"event": "shown"|"accepted", "id", "user"}  (appended to --telemetry-log)
GET  /health
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from codecomplete.generation import Completer, load_model_and_tokenizer


class CompleteRequest(BaseModel):
    prefix: str
    suffix: str = ""
    mode: str = "line"
    max_new_tokens: int | None = None


class TelemetryEvent(BaseModel):
    event: str
    id: str
    user: str
    language: str = ""
    chars: int = 0


def create_app(completer: Completer, telemetry_log: Path, cache_size: int = 256) -> FastAPI:
    app = FastAPI()
    gpu_lock = threading.Lock()  # one generation at a time; requests queue behind it
    cache: OrderedDict[tuple, str] = OrderedDict()
    log_lock = threading.Lock()

    def run(req: CompleteRequest):
        # The cursor context near the edit point decides the completion; key the cache on it.
        key = (req.prefix[-2000:], req.suffix[:500], req.mode)
        if key in cache:
            cache.move_to_end(key)
            return cache[key], 0.0, True
        with gpu_lock:
            c = completer.complete(req.prefix, req.suffix, req.mode, req.max_new_tokens)
        cache[key] = c.text
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return c.text, c.latency_ms, False

    @app.post("/complete")
    async def complete(req: CompleteRequest):
        text, latency, cached = await run_in_threadpool(run, req)
        return {"id": uuid.uuid4().hex, "text": text, "latency_ms": round(latency, 1), "cached": cached}

    @app.post("/telemetry")
    async def telemetry(e: TelemetryEvent):
        with log_lock, telemetry_log.open("a") as f:
            f.write(json.dumps({**e.model_dump(), "ts": time.time()}) + "\n")
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/merged")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--telemetry-log", default="data/telemetry.jsonl")
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter, load_4bit=args.load_4bit)
    completer = Completer(model, tokenizer)
    for _ in range(3):  # warm up CUDA kernels so the first user request isn't slow
        completer.complete("def add(a, b):\n    ", mode="line")
    log = Path(args.telemetry_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(completer, log), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

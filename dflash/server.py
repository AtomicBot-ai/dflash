"""
DFlash MLX Inference Server — OpenAI-compatible HTTP API.

Supports both standard mlx-lm inference and DFlash-accelerated
speculative decoding when a --draft-model is provided.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import mlx.core as mx
import uvicorn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="[mlx] %(message)s")
log = logging.getLogger("dflash.server")

_model = None
_draft = None
_tokenizer: Optional[TokenizerWrapper] = None
_model_id: str = ""
_api_key: str = ""
_ctx_size: int = 4096
_block_size: int = 0
_cancel_event: asyncio.Event = asyncio.Event()


def _generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _ts() -> int:
    return int(time.time())


def _extract_text(messages: list[dict]) -> str:
    """Apply the tokenizer chat template to produce a prompt string."""
    if _tokenizer is None:
        raise RuntimeError("tokenizer not loaded")
    tok = _tokenizer._tokenizer
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts: list[str] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        parts.append(f"{m['role']}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _models(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _model_id,
                    "object": "model",
                    "created": _ts(),
                    "owned_by": "mlx",
                }
            ],
        }
    )


def _check_auth(request: Request) -> Optional[JSONResponse]:
    if not _api_key:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {_api_key}":
        return JSONResponse(
            {
                "error": {
                    "message": "Unauthorized",
                    "type": "authentication_error",
                    "code": "unauthorized",
                }
            },
            status_code=401,
        )
    return None


_SENTINEL = object()


def _do_generate(
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    stop: list[str],
) -> AsyncGenerator[dict[str, Any], None]:
    """Run sync MLX generation in a thread, bridged to an async generator via Queue."""

    async def _inner() -> AsyncGenerator[dict[str, Any], None]:
        _cancel_event.clear()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()

        def _put(item: dict[str, Any] | object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def _sync_worker() -> None:
            try:
                prompt_tokens = _tokenizer.encode(prompt_text)  # type: ignore[union-attr]
                prompt_size = len(prompt_tokens)
                prompt_array = mx.array(prompt_tokens)

                gen_tokens: list[int] = []
                text_so_far = ""
                tic = time.perf_counter()

                if _draft is not None:
                    from dflash.model_mlx import stream_generate as dflash_stream
                    from mlx_lm.sample_utils import make_sampler

                    sampler = make_sampler(temp=temperature)
                    bs = _block_size if _block_size > 0 else None

                    for resp in dflash_stream(
                        _model,
                        _draft,
                        _tokenizer,
                        prompt_array,
                        block_size=bs,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        sampler=sampler,
                    ):
                        if _cancel_event.is_set():
                            break
                        delta_text = resp.text
                        if delta_text:
                            text_so_far += delta_text
                            gen_tokens.extend(resp.tokens)
                            _put({"type": "chunk", "content": delta_text})

                        if resp.finish_reason is not None:
                            elapsed = time.perf_counter() - tic
                            _put({
                                "type": "done",
                                "finish_reason": resp.finish_reason,
                                "usage": {
                                    "prompt_tokens": prompt_size,
                                    "completion_tokens": len(gen_tokens),
                                    "total_tokens": prompt_size + len(gen_tokens),
                                },
                                "timings": {
                                    "prompt_n": prompt_size,
                                    "predicted_n": len(gen_tokens),
                                    "predicted_per_second": (
                                        len(gen_tokens) / elapsed if elapsed > 0 else 0
                                    ),
                                    "prompt_per_second": resp.prompt_tps,
                                },
                            })
                            return
                else:
                    from mlx_lm import stream_generate as mlx_stream
                    from mlx_lm.sample_utils import make_sampler

                    sampler = make_sampler(
                        temp=temperature,
                        top_p=1.0,
                        min_tokens_to_keep=1,
                    )

                    for resp in mlx_stream(
                        _model,
                        _tokenizer,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        sampler=sampler,
                    ):
                        if _cancel_event.is_set():
                            break
                        token_text = resp.text
                        if token_text:
                            gen_tokens.append(0)
                            text_so_far += token_text
                            _put({"type": "chunk", "content": token_text})

                            should_stop = False
                            for s in stop:
                                if s and s in text_so_far:
                                    should_stop = True
                                    break
                            if should_stop:
                                break

                    elapsed = time.perf_counter() - tic
                    _put({
                        "type": "done",
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": prompt_size,
                            "completion_tokens": len(gen_tokens),
                            "total_tokens": prompt_size + len(gen_tokens),
                        },
                        "timings": {
                            "prompt_n": prompt_size,
                            "predicted_n": len(gen_tokens),
                            "predicted_per_second": (
                                len(gen_tokens) / elapsed if elapsed > 0 else 0
                            ),
                            "prompt_per_second": (
                                prompt_size / elapsed if elapsed > 0 else 0
                            ),
                        },
                    })
            except Exception as exc:
                log.exception("Generation error")
                _put({"type": "error", "message": str(exc)})
            finally:
                _put(_SENTINEL)

        loop.run_in_executor(None, _sync_worker)

        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            if isinstance(event, dict) and event.get("type") == "error":
                log.error("Generation failed: %s", event["message"])
                break
            yield event  # type: ignore[misc]

    return _inner()


async def _chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error", "code": None}},
            status_code=400,
        )

    messages = body.get("messages", [])
    temperature = float(body.get("temperature", 0.7))
    max_tokens = body.get("max_tokens") or body.get("n_predict") or 2048
    max_tokens = int(max_tokens)
    repetition_penalty = float(body.get("repetition_penalty", 1.0))
    stop_sequences: list[str] = body.get("stop", []) or []
    is_streaming = bool(body.get("stream", False))
    model_name = body.get("model", _model_id)

    log.info(
        "Request: model=%s, messages=%d, stream=%s",
        model_name,
        len(messages),
        is_streaming,
    )

    prompt_text = _extract_text(messages)

    if is_streaming:
        return await _handle_streaming(
            prompt_text, model_name, max_tokens, temperature,
            repetition_penalty, stop_sequences,
        )
    else:
        return await _handle_non_streaming(
            prompt_text, model_name, max_tokens, temperature,
            repetition_penalty, stop_sequences,
        )


async def _handle_non_streaming(
    prompt_text: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    stop: list[str],
) -> JSONResponse:
    full_text = ""
    usage_info: dict[str, Any] = {}

    gen = _do_generate(prompt_text, max_tokens, temperature, repetition_penalty, stop)
    async for event in gen:
        if event["type"] == "chunk":
            full_text += event["content"]
        elif event["type"] == "done":
            usage_info = event.get("usage", {})

    response_id = _generate_id()
    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": _ts(),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_info,
        }
    )


async def _handle_streaming(
    prompt_text: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    stop: list[str],
) -> StreamingResponse:
    response_id = _generate_id()
    created = _ts()

    async def _event_stream() -> AsyncGenerator[str, None]:
        initial_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        gen = _do_generate(
            prompt_text, max_tokens, temperature, repetition_penalty, stop
        )
        async for event in gen:
            if event["type"] == "chunk":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event["content"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event["type"] == "done":
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": event.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": event.get("usage"),
                    "timings": event.get("timings"),
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _cancel(_request: Request) -> JSONResponse:
    _cancel_event.set()
    log.info("Cancel requested")
    return JSONResponse({"status": "cancelled", "cancelled_count": 1})


def _load_models(args: argparse.Namespace) -> None:
    global _model, _draft, _tokenizer, _model_id, _api_key, _ctx_size, _block_size

    from pathlib import Path

    from mlx_lm import load as mlx_load

    model_path = Path(args.model)
    if model_path.is_file():
        model_path = model_path.parent

    log.info("Loading target model: %s", model_path)
    _model, _tokenizer = mlx_load(str(model_path))
    log.info("Target model loaded successfully")

    if args.draft_model:
        from dflash.model_mlx import load_draft

        log.info("Loading DFlash draft model: %s", args.draft_model)
        _draft = load_draft(args.draft_model)
        log.info("DFlash draft model loaded (block_size=%d)", _draft.config.block_size)

    _model_id = args.model_id or model_path.stem
    _api_key = args.api_key or ""
    _ctx_size = args.ctx_size
    _block_size = args.block_size


def create_app() -> Starlette:
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/v1/models", _models, methods=["GET"]),
        Route("/v1/chat/completions", _chat_completions, methods=["POST"]),
        Route("/v1/cancel", _cancel, methods=["POST"]),
    ]
    return Starlette(routes=routes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DFlash MLX Server")
    parser.add_argument(
        "--model", "-m", required=True, help="Path to the MLX model directory"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context window size")
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument(
        "--model-id", default="", help="Model ID reported by the API"
    )
    parser.add_argument(
        "--draft-model",
        default="",
        help="Path to DFlash draft model (enables speculative decoding)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="DFlash block size (0 = use model default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_models(args)

    app = create_app()

    log.info("http server listening on http://127.0.0.1:%d", args.port)
    log.info("server is listening on 127.0.0.1:%d", args.port)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

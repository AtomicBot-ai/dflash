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
import re
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


def _extract_text(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> str:
    """Apply the tokenizer chat template to produce a prompt string.

    When `tools` are provided, they are forwarded to `apply_chat_template` so
    the model sees tool schemas rendered per its template (Qwen, Hermes,
    Llama 3.1, Mistral, etc.). Falls back gracefully for tokenizers that do
    not accept the `tools` kwarg.
    """
    if _tokenizer is None:
        raise RuntimeError("tokenizer not loaded")
    tok = _tokenizer._tokenizer
    if hasattr(tok, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        try:
            rendered = tok.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            log.warning(
                "tokenizer.apply_chat_template does not accept tools kwarg; "
                "falling back without tool schema"
            )
            rendered = tok.apply_chat_template(messages, **kwargs)
        if tools:
            has_tool_section = (
                "<tools>" in rendered
                or "tool_call" in rendered
                or "function" in rendered.lower()
            )
            log.info(
                "prompt rendered: len=%d, tools_in_prompt=%s",
                len(rendered),
                has_tool_section,
            )
            if not has_tool_section:
                log.warning(
                    "tools were passed but NOT rendered into the prompt — "
                    "chat_template likely ignores the `tools` variable. "
                    "Prompt tail: ...%s",
                    rendered[-400:],
                )
        return rendered
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


_TOOL_CALL_START_MARKERS: tuple[str, ...] = (
    "<tool_call>",
    "<|tool_call|>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
)

_TOOL_CALL_MAX_MARKER_LEN = max(len(m) for m in _TOOL_CALL_START_MARKERS)

_TOOL_CALL_QWEN_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_HERMES_PIPE_RE = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})(?=\s*(?:<\||$))", re.DOTALL
)
_TOOL_CALL_LLAMA_RE = re.compile(
    r"<\|python_tag\|>\s*(\{.*?\})(?:\s*<\|eom_id\|>|\s*$)", re.DOTALL
)
_TOOL_CALL_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL)

_TOOL_CALL_QWEN_XML_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_CALL_QWEN_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _to_tool_call(obj: dict) -> Optional[dict]:
    """Coerce a model-produced dict into OpenAI tool_call shape."""
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = obj.get("name") or (fn.get("name") if fn else None)
    if not name:
        return None
    args: Any = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None and fn is not None:
        args = fn.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, str):
        try:
            args = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            args = "{}"
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": str(name), "arguments": args},
    }


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Detect tool-call spans produced by common templates and extract them.

    Returns `(cleaned_content, tool_calls)`. If no tool calls are found, the
    original text is returned unchanged and `tool_calls` is empty.
    """
    if not text:
        return text, []

    calls: list[dict] = []

    matches = list(_TOOL_CALL_QWEN_XML_RE.finditer(text))
    if matches:
        for m in matches:
            fn_name = m.group(1).strip()
            inner = m.group(2)
            args_dict: dict[str, Any] = {}
            for pm in _TOOL_CALL_QWEN_XML_PARAM_RE.finditer(inner):
                key = pm.group(1).strip()
                raw_val = pm.group(2).strip()
                parsed: Any = raw_val
                try:
                    parsed = json.loads(raw_val)
                except (json.JSONDecodeError, ValueError):
                    pass
                args_dict[key] = parsed
            call = _to_tool_call({"name": fn_name, "arguments": args_dict})
            if call:
                calls.append(call)
        if calls:
            cleaned = _TOOL_CALL_QWEN_XML_RE.sub("", text).strip()
            return cleaned, calls

    matches = list(_TOOL_CALL_QWEN_RE.finditer(text))
    if matches:
        for m in matches:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            call = _to_tool_call(obj)
            if call:
                calls.append(call)
        if calls:
            cleaned = _TOOL_CALL_QWEN_RE.sub("", text).strip()
            return cleaned, calls

    matches = list(_TOOL_CALL_HERMES_PIPE_RE.finditer(text))
    if matches:
        for m in matches:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            call = _to_tool_call(obj)
            if call:
                calls.append(call)
        if calls:
            cleaned = _TOOL_CALL_HERMES_PIPE_RE.sub("", text).strip()
            return cleaned, calls

    m = _TOOL_CALL_LLAMA_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            call = _to_tool_call(obj)
            if call:
                calls.append(call)
                cleaned = text[: m.start()].strip()
                return cleaned, calls
        except json.JSONDecodeError:
            pass

    m = _TOOL_CALL_MISTRAL_RE.search(text)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                for obj in arr:
                    call = _to_tool_call(obj)
                    if call:
                        calls.append(call)
                if calls:
                    cleaned = text[: m.start()].strip()
                    return cleaned, calls
        except json.JSONDecodeError:
            pass

    return text, []


def _has_tool_call_marker(text: str) -> bool:
    return any(marker in text for marker in _TOOL_CALL_START_MARKERS)


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

                # Qwen3-style templates add <think> to the generation prompt,
                # so the model output starts with reasoning directly.
                # Echo the tag so the frontend middleware can detect it.
                _stripped = prompt_text.rstrip("\n")
                if _stripped.endswith("<think>") and not _stripped.endswith("</think>"):
                    _put({"type": "chunk", "content": "<think>\n"})

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
                            _put(
                                {
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
                                            len(gen_tokens) / elapsed
                                            if elapsed > 0
                                            else 0
                                        ),
                                        "prompt_per_second": resp.prompt_tps,
                                    },
                                }
                            )
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
                    _put(
                        {
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
                        }
                    )
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
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
            status_code=400,
        )

    messages = body.get("messages", [])
    temperature = float(body.get("temperature", 0.7))
    max_tokens = (
        body.get("max_tokens")
        or body.get("max_output_tokens")
        or body.get("n_predict")
        or 16384
    )
    max_tokens = int(max_tokens)
    repetition_penalty = float(body.get("repetition_penalty", 1.0))
    stop_sequences: list[str] = body.get("stop", []) or []
    is_streaming = bool(body.get("stream", False))
    model_name = body.get("model", _model_id)
    tools = body.get("tools") or None
    tool_choice = body.get("tool_choice")

    log.info(
        "Request: model=%s, messages=%d, stream=%s, tools=%d",
        model_name,
        len(messages),
        is_streaming,
        len(tools) if isinstance(tools, list) else 0,
    )

    prompt_text = _extract_text(messages, tools=tools, tool_choice=tool_choice)
    tools_enabled = bool(tools)

    if is_streaming:
        return await _handle_streaming(
            prompt_text,
            model_name,
            max_tokens,
            temperature,
            repetition_penalty,
            stop_sequences,
            tools_enabled,
        )
    else:
        return await _handle_non_streaming(
            prompt_text,
            model_name,
            max_tokens,
            temperature,
            repetition_penalty,
            stop_sequences,
            tools_enabled,
        )


async def _handle_non_streaming(
    prompt_text: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    stop: list[str],
    tools_enabled: bool = False,
) -> JSONResponse:
    full_text = ""
    usage_info: dict[str, Any] = {}

    gen = _do_generate(prompt_text, max_tokens, temperature, repetition_penalty, stop)
    async for event in gen:
        if event["type"] == "chunk":
            full_text += event["content"]
        elif event["type"] == "done":
            usage_info = event.get("usage", {})

    tool_calls: list[dict] = []
    content: Optional[str] = full_text
    finish_reason = "stop"
    if tools_enabled:
        log.info(
            "raw model output (tools_enabled=True, len=%d): %r",
            len(full_text),
            full_text[:1200] + ("...[truncated]" if len(full_text) > 1200 else ""),
        )
        if _has_tool_call_marker(full_text):
            cleaned, tool_calls = _parse_tool_calls(full_text)
            if tool_calls:
                content = cleaned or None
                finish_reason = "tool_calls"
        else:
            log.warning(
                "tools_enabled but model produced no <tool_call>/<|python_tag|>/"
                "[TOOL_CALLS] markers — model did not follow tool-calling format"
            )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

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
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_info,
        }
    )


def _safe_emit_split(accumulated_unsent: str) -> tuple[str, str]:
    """Split unsent text into (safe_to_emit, hold_back) such that `hold_back`
    keeps any trailing partial prefix of a tool-call start marker intact.

    Without this, we could leak the first characters of e.g. "<tool_call>" to
    the client before realizing a tool call began, making downstream parsing
    inconsistent.
    """
    if not accumulated_unsent:
        return "", ""
    hold = min(len(accumulated_unsent), _TOOL_CALL_MAX_MARKER_LEN)
    for i in range(hold, 0, -1):
        tail = accumulated_unsent[-i:]
        if any(marker.startswith(tail) for marker in _TOOL_CALL_START_MARKERS):
            return accumulated_unsent[:-i], tail
    return accumulated_unsent, ""


async def _handle_streaming(
    prompt_text: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    stop: list[str],
    tools_enabled: bool = False,
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

        def _content_chunk(content: str) -> str:
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        full_text = ""
        unsent = ""
        tool_call_detected = False

        gen = _do_generate(
            prompt_text, max_tokens, temperature, repetition_penalty, stop
        )
        async for event in gen:
            if event["type"] == "chunk":
                piece = event["content"]
                full_text += piece

                if not tools_enabled:
                    yield _content_chunk(piece)
                    continue

                if tool_call_detected:
                    continue

                unsent += piece
                if _has_tool_call_marker(full_text):
                    tool_call_detected = True
                    unsent = ""
                    continue

                safe, unsent = _safe_emit_split(unsent)
                if safe:
                    yield _content_chunk(safe)

            elif event["type"] == "done":
                finish_reason = event.get("finish_reason", "stop")
                tool_calls: list[dict] = []

                if tools_enabled:
                    log.info(
                        "raw model output (stream, tools_enabled=True, len=%d): %r",
                        len(full_text),
                        full_text[:1200]
                        + ("...[truncated]" if len(full_text) > 1200 else ""),
                    )
                    if _has_tool_call_marker(full_text):
                        _cleaned, tool_calls = _parse_tool_calls(full_text)
                    else:
                        log.warning(
                            "tools_enabled but stream produced no tool-call "
                            "markers — model did not follow tool-calling format"
                        )

                if tool_calls:
                    for idx, call in enumerate(tool_calls):
                        delta_call = {
                            "index": idx,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        tc_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [delta_call]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"
                    finish_reason = "tool_calls"
                elif tools_enabled and unsent:
                    yield _content_chunk(unsent)
                    unsent = ""

                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
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
    parser.add_argument(
        "--ctx-size", type=int, default=4096, help="Context window size"
    )
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument("--model-id", default="", help="Model ID reported by the API")
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

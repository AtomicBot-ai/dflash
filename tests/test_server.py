"""
Integration tests for the DFlash MLX server.

Two modes:
  1. Mock mode (default): patches generation, runs without a real model.
     Validates endpoint contracts, SSE format, auth, and error handling.
  2. Live mode: set MLX_TEST_MODEL=/path/to/model to run against real MLX inference.
     Validates end-to-end generation including streaming.

Run:
    python -m pytest tests/test_server.py -v
    MLX_TEST_MODEL=~/models/Qwen3.5-9B-MLX-4bit python -m pytest tests/test_server.py -v -k live
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

import dflash.server as srv
from dflash.server import create_app

TEST_MODEL_ID = "test-model"
TEST_API_KEY = "test-secret-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def mock_app() -> AsyncGenerator[httpx.AsyncClient, None]:
    """ASGI test client with mocked model globals (no real MLX needed)."""
    _orig = (srv._model, srv._tokenizer, srv._model_id, srv._api_key, srv._draft)

    srv._model = MagicMock()
    srv._draft = None
    srv._model_id = TEST_MODEL_ID
    srv._api_key = TEST_API_KEY
    srv._tokenizer = _make_fake_tokenizer()

    app = create_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    srv._model, srv._tokenizer, srv._model_id, srv._api_key, srv._draft = _orig


@pytest_asyncio.fixture
async def mock_app_no_auth() -> AsyncGenerator[httpx.AsyncClient, None]:
    """ASGI test client with no api-key requirement."""
    _orig = (srv._model, srv._tokenizer, srv._model_id, srv._api_key, srv._draft)

    srv._model = MagicMock()
    srv._draft = None
    srv._model_id = TEST_MODEL_ID
    srv._api_key = ""
    srv._tokenizer = _make_fake_tokenizer()

    app = create_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    srv._model, srv._tokenizer, srv._model_id, srv._api_key, srv._draft = _orig


def _make_fake_tokenizer() -> MagicMock:
    """Create a tokenizer mock that returns deterministic tokens."""
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    inner = MagicMock()
    inner.apply_chat_template.return_value = "user: Hello\nassistant:"
    tok._tokenizer = inner
    return tok


def _auth_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEST_API_KEY}",
    }


def _chat_body(stream: bool = False) -> dict[str, Any]:
    return {
        "model": TEST_MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": stream,
        "max_tokens": 32,
        "temperature": 0.1,
    }


# ---------------------------------------------------------------------------
# Mock generation patch — replaces _do_generate for predictable output
# ---------------------------------------------------------------------------


def _patch_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _do_generate with a fake that yields known tokens."""

    def _fake_generate(
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
        stop: list[str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def _inner() -> AsyncGenerator[dict[str, Any], None]:
            for word in ["Hello", " world", "!"]:
                yield {"type": "chunk", "content": word}
            yield {
                "type": "done",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 3,
                    "total_tokens": 6,
                },
                "timings": {
                    "prompt_n": 3,
                    "predicted_n": 3,
                    "predicted_per_second": 30.0,
                    "prompt_per_second": 100.0,
                },
            }

        return _inner()

    monkeypatch.setattr(srv, "_do_generate", _fake_generate)


# ===========================================================================
# Mock tests — no real model needed
# ===========================================================================


@pytest.mark.anyio
async def test_health(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_models(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.get(
        "/v1/models", headers={"Authorization": f"Bearer {TEST_API_KEY}"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == TEST_MODEL_ID
    assert data["data"][0]["object"] == "model"


@pytest.mark.anyio
async def test_auth_rejected_without_key(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.post(
        "/v1/chat/completions",
        json=_chat_body(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "unauthorized"


@pytest.mark.anyio
async def test_auth_rejected_wrong_key(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.post(
        "/v1/chat/completions",
        json=_chat_body(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer wrong-key",
        },
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_auth_not_required_when_empty(
    mock_app_no_auth: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_generate(monkeypatch)
    resp = await mock_app_no_auth.post(
        "/v1/chat/completions",
        json=_chat_body(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_cancel(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.post("/v1/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"


@pytest.mark.anyio
async def test_chat_non_streaming(
    mock_app: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_generate(monkeypatch)
    resp = await mock_app.post(
        "/v1/chat/completions",
        json=_chat_body(stream=False),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert "id" in data
    assert data["model"] == TEST_MODEL_ID
    assert len(data["choices"]) == 1

    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world!"
    assert choice["finish_reason"] == "stop"

    assert data["usage"]["prompt_tokens"] == 3
    assert data["usage"]["completion_tokens"] == 3
    assert data["usage"]["total_tokens"] == 6


@pytest.mark.anyio
async def test_chat_streaming(
    mock_app: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_generate(monkeypatch)
    resp = await mock_app.post(
        "/v1/chat/completions",
        json=_chat_body(stream=True),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]

    assert len(data_lines) >= 3, f"Expected at least 3 data lines, got {len(data_lines)}: {data_lines}"

    # First chunk: role
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["finish_reason"] is None

    # Content chunks
    content_chunks = []
    for dl in data_lines[1:]:
        payload = dl.removeprefix("data: ")
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        assert chunk["object"] == "chat.completion.chunk"
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            content_chunks.append(delta["content"])

    assert "".join(content_chunks) == "Hello world!"

    # Final chunk has finish_reason
    final_data = [
        json.loads(dl.removeprefix("data: "))
        for dl in data_lines
        if dl != "data: [DONE]"
    ]
    last_chunk = final_data[-1]
    assert last_chunk["choices"][0]["finish_reason"] == "stop"
    assert "usage" in last_chunk

    # [DONE] sentinel
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.anyio
async def test_chat_streaming_sse_format(
    mock_app: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate raw SSE wire format: each event is 'data: ...' followed by blank line."""
    _patch_generate(monkeypatch)
    resp = await mock_app.post(
        "/v1/chat/completions",
        json=_chat_body(stream=True),
        headers=_auth_headers(),
    )
    raw = resp.text
    # Every SSE event must end with \n\n
    events = raw.split("\n\n")
    non_empty = [e.strip() for e in events if e.strip()]
    for event in non_empty:
        assert event.startswith("data: "), f"SSE event must start with 'data: ': {event!r}"


@pytest.mark.anyio
async def test_chat_invalid_json_body(mock_app: httpx.AsyncClient) -> None:
    resp = await mock_app.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TEST_API_KEY}",
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data


# ===========================================================================
# Live integration tests — require MLX_TEST_MODEL env var
# ===========================================================================

LIVE_MODEL = os.environ.get("MLX_TEST_MODEL", "")
live = pytest.mark.skipif(not LIVE_MODEL, reason="MLX_TEST_MODEL not set")


@pytest.fixture(scope="module")
def live_server_url() -> str:
    """Start a real DFlash server in a subprocess for live testing."""
    if not LIVE_MODEL:
        pytest.skip("MLX_TEST_MODEL not set")

    import subprocess
    import time

    port = 19876
    proc = subprocess.Popen(
        [
            "python", "-m", "dflash.server",
            "--model", LIVE_MODEL,
            "--port", str(port),
            "--model-id", "live-test-model",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"Server exited early (code {proc.returncode}): {out}")
        line = proc.stdout.readline() if proc.stdout else ""
        if "server is listening" in line.lower():
            ready = True
            break

    if not ready:
        proc.kill()
        pytest.fail("Server did not become ready within 120s")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@live
def test_live_health(live_server_url: str) -> None:
    resp = httpx.get(f"{live_server_url}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@live
def test_live_models(live_server_url: str) -> None:
    resp = httpx.get(f"{live_server_url}/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


@live
def test_live_chat_non_streaming(live_server_url: str) -> None:
    resp = httpx.post(
        f"{live_server_url}/v1/chat/completions",
        json={
            "model": "live-test-model",
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "stream": False,
            "max_tokens": 16,
            "temperature": 0.0,
        },
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Expected non-empty response"
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert "usage" in data


@live
def test_live_chat_streaming(live_server_url: str) -> None:
    with httpx.stream(
        "POST",
        f"{live_server_url}/v1/chat/completions",
        json={
            "model": "live-test-model",
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "stream": True,
            "max_tokens": 16,
            "temperature": 0.0,
        },
        headers={"Content-Type": "application/json"},
        timeout=60,
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        chunks: list[str] = []
        saw_done = False
        saw_role = False
        saw_finish = False

        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                saw_done = True
                continue
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"
                delta = chunk["choices"][0].get("delta", {})
                if "role" in delta:
                    saw_role = True
                if "content" in delta:
                    chunks.append(delta["content"])
                if chunk["choices"][0].get("finish_reason") is not None:
                    saw_finish = True

        assert saw_role, "Missing initial role chunk"
        assert saw_finish, "Missing finish_reason chunk"
        assert saw_done, "Missing [DONE] sentinel"
        full_text = "".join(chunks)
        assert len(full_text) > 0, "Expected non-empty streamed content"

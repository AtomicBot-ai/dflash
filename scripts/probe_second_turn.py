"""Reproduce the `apply_chat_template` crash that happened in-app after the
first successful tool call, then verify `_normalize_messages_for_template`
fixes it. Run:

    uv run --with transformers --with jinja2 --with sentencepiece --with tiktoken \
        python scripts/probe_second_turn.py \
        "/Users/misha/Library/Application Support/Atomic Chat/data/mlx/models/Qwen3.5-9B-MLX-4bit"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue
        if msg.get("role") == "assistant" and isinstance(
            msg.get("tool_calls"), list
        ):
            new_msg = dict(msg)
            new_calls: list[dict] = []
            for call in msg["tool_calls"]:
                if not isinstance(call, dict):
                    new_calls.append(call)
                    continue
                new_call = dict(call)
                fn = new_call.get("function")
                if isinstance(fn, dict):
                    new_fn = dict(fn)
                    args = new_fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            new_fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            new_fn["arguments"] = {}
                    new_call["function"] = new_fn
                new_calls.append(new_call)
            new_msg["tool_calls"] = new_calls
            normalized.append(new_msg)
        else:
            normalized.append(msg)
    return normalized



def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_dir>", file=sys.stderr)
        return 2
    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search_exa",
                "description": "Search the web via Exa.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    # Second-turn history mirroring what Atomic-Chat sends after the first
    # successful tool call (arguments as JSON string per OpenAI spec).
    messages = [
        {"role": "user", "content": "Какие праздники 16 апреля 2026?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "web_search_exa",
                        "arguments": '{"query": "holidays April 16 2026"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '[{"title": "World Voice Day", "date": "2026-04-16"}]',
        },
    ]

    print("===== RAW (arguments as string, as sent by frontend) =====")
    try:
        out = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        print(out[-500:])
        print("\n  RESULT: template rendered — no crash.")
    except Exception as exc:
        print(f"  CRASH (expected): {type(exc).__name__}: {exc}")

    print("\n===== NORMALIZED (arguments coerced to dict) =====")
    normalized = _normalize_messages_for_template(messages)
    out = tok.apply_chat_template(
        normalized, tools=tools, tokenize=False, add_generation_prompt=True
    )
    print(out[-500:])
    print(f"\n  RESULT: template rendered OK, length={len(out)}")

    # Dump the JSON for clarity
    print("\n===== JSON of rendered second-turn prompt =====")
    print(json.dumps(out[-800:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

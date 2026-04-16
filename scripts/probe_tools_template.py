"""Diagnose whether a Qwen-style tokenizer's chat_template renders `tools`.

Run:
    python3 scripts/probe_tools_template.py \
        "/Users/misha/Library/Application Support/Atomic Chat/data/mlx/models/Qwen3.5-9B-MLX-4bit"

If you see `<tools>` / `"function"` markers in the printed prompt, the template
honors the `tools=` kwarg. If not, `_extract_text` needs to inject the tool
schema manually (Qwen-style system block) before calling `apply_chat_template`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_dir>", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1]).expanduser()
    if not model_dir.is_dir():
        print(f"Not a directory: {model_dir}", file=sys.stderr)
        return 2

    print(f"Loading tokenizer from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    messages = [
        {"role": "user", "content": "Какие праздники 16 апреля 2026?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search_exa",
                "description": "Search the web via Exa.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    print("\n===== BASELINE (no tools) =====")
    baseline = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(baseline)
    print(f"\n[len={len(baseline)}]")

    print("\n===== WITH tools=[...] =====")
    try:
        rendered = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as exc:
        print(f"FAIL: apply_chat_template rejected tools kwarg: {exc}")
        return 1

    print(rendered)
    print(f"\n[len={len(rendered)}]")

    markers = {
        "<tools>": "<tools>" in rendered,
        "</tools>": "</tools>" in rendered,
        "<tool_call>": "<tool_call>" in rendered,
        '"function"': '"function"' in rendered,
        "web_search_exa": "web_search_exa" in rendered,
        "# Tools": "# Tools" in rendered,
    }
    print("\n===== DIAGNOSIS =====")
    for k, v in markers.items():
        print(f"  {k!r:<18} -> {v}")

    diff = len(rendered) - len(baseline)
    print(f"\n  prompt grew by {diff} chars when tools were added")

    if rendered == baseline:
        print(
            "\n  VERDICT: HERESY CONFIRMED — chat_template ignores the `tools` "
            "variable. Must inject tool schema manually into a system message."
        )
        return 3
    if not markers["web_search_exa"]:
        print(
            "\n  VERDICT: Prompt differs but tool name is NOT embedded. Template "
            "likely references `tools` but in an unexpected way."
        )
        return 4
    print(
        "\n  VERDICT: Tools ARE rendered into the prompt. Ересь №1 не "
        "подтверждена — нужно смотреть сам вывод модели."
    )
    # Also print a JSON dump of the rendered prompt so tokens are unambiguous.
    print("\n===== JSON-DUMP OF RENDERED PROMPT =====")
    print(json.dumps(rendered))
    return 0


if __name__ == "__main__":
    sys.exit(main())

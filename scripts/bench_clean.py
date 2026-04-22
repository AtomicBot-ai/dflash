"""Clean DFlash vs baseline benchmark — no HTTP, no Tauri, pure MLX.

Measures, for a fixed target + draft pair, on identical prompts with
``temperature=0`` (greedy):

* baseline tok/s (plain ``mlx_lm.stream_generate``)
* DFlash tok/s (``dflash.model_mlx.stream_generate``)
* average acceptance rate per block
* peak unified memory

Run:

    cd /Users/misha/Work/Atomic/dflash
    .venv/bin/python scripts/bench_clean.py
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import stream_generate as mlx_stream_generate
from mlx_lm.sample_utils import make_sampler

from dflash.model_mlx import load as dflash_load
from dflash.model_mlx import load_draft
from dflash.model_mlx import stream_generate as dflash_stream_generate

DEFAULT_TARGET = "/Users/misha/Library/Application Support/Atomic Chat/data/mlx/models/Qwen3.5-4B-MLX-4bit"
DEFAULT_DRAFT = "/Users/misha/Library/Application Support/Atomic Chat/data/mlx/models/z-lab_Qwen3.5-4B-DFlash"


PROMPTS: dict[str, str] = {
    "math-divisors": (
        "How many positive whole-number divisors does 196 have?\n"
        "Think step by step and show all reasoning."
    ),
    "logic-5person": (
        "Read the following information carefully and answer the questions given below:\n"
        "i.  There is a group of five persons A, B, C, D and E.\n"
        "ii. One of them is a horticulturist, one is a physicist, one is a journalist,\n"
        "    one is an industrialist and one is an advocate.\n"
        "iii. Three of them A, C and advocate prefer tea to coffee and two of them - B\n"
        "     and the journalist prefer coffee to tea.\n"
        "iv. The industrialist and D and A are friends to one another but two of them\n"
        "    prefer coffee to tea.\n"
        "v.  The horticulturist is C's brother.\n\n"
        "What are the professions for A, B, C, D, E? Show your reasoning."
    ),
    "code-sliding-window": (
        "Write a Python function `longest_subarray_with_sum_at_most_k(nums, k)` that, "
        "given a list of integers and an integer k, returns the length of the longest "
        "contiguous subarray whose sum is <= k. Use a sliding-window approach. Explain "
        "your algorithm step by step before coding. Include edge cases, complexity "
        "analysis, and 5 test cases with expected output."
    ),
}


@dataclass
class Result:
    name: str
    mode: str
    tok_s: float
    tokens: int
    elapsed_s: float
    peak_mem_gb: float
    avg_accepted: float | None
    finish_reason: str | None


def _render_prompt(tokenizer: Any, user_text: str, enable_thinking: bool) -> str:
    tok = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def _bench_baseline(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
) -> Result:
    sampler = make_sampler(temp=temperature)
    mx.reset_peak_memory()

    n_tokens = 0
    finish_reason = None
    tic = time.perf_counter()
    for resp in mlx_stream_generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        n_tokens += 1
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            break
    elapsed = time.perf_counter() - tic
    peak = mx.get_peak_memory() / 1e9
    return Result(
        name="baseline",
        mode="plain-mlx",
        tok_s=n_tokens / elapsed if elapsed > 0 else 0.0,
        tokens=n_tokens,
        elapsed_s=elapsed,
        peak_mem_gb=peak,
        avg_accepted=None,
        finish_reason=finish_reason,
    )


def _bench_dflash(
    model: Any,
    draft: Any,
    tokenizer: Any,
    prompt_text: str,
    max_tokens: int,
    block_size: int,
    temperature: float,
) -> Result:
    sampler = make_sampler(temp=temperature)
    mx.reset_peak_memory()

    n_tokens = 0
    accepted_sum = 0
    block_count = 0
    finish_reason = None
    tic = time.perf_counter()
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_arr = mx.array(prompt_ids)

    for resp in dflash_stream_generate(
        model,
        draft,
        tokenizer,
        prompt_arr,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temperature,
        sampler=sampler,
    ):
        if resp.tokens:
            n_tokens += len(resp.tokens)
            accepted_sum += resp.accepted
            block_count += 1
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            break
    elapsed = time.perf_counter() - tic
    peak = mx.get_peak_memory() / 1e9
    avg_acc = accepted_sum / block_count if block_count else 0.0
    return Result(
        name="dflash",
        mode=f"dflash-b{block_size}",
        tok_s=n_tokens / elapsed if elapsed > 0 else 0.0,
        tokens=n_tokens,
        elapsed_s=elapsed,
        peak_mem_gb=peak,
        avg_accepted=avg_acc,
        finish_reason=finish_reason,
    )


def _warmup(run: Callable[[], Any]) -> None:
    print("  warmup…", flush=True)
    try:
        run()
    except Exception as e:
        print(f"  warmup failed: {e!r}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean DFlash vs baseline MLX benchmark.")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target model path or HF id.")
    p.add_argument("--draft", default=DEFAULT_DRAFT, help="DFlash draft model path or HF id.")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Qwen thinking mode (enable_thinking=False).",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        choices=list(PROMPTS.keys()),
        help="Subset of prompts to run (default: all).",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only run DFlash path.",
    )
    p.add_argument(
        "--skip-dflash",
        action="store_true",
        help="Only run baseline path.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    target_path = args.target
    draft_path = args.draft
    max_tokens = args.max_tokens
    block_size = args.block_size
    temperature = args.temperature
    enable_thinking = not args.no_think
    prompt_keys = args.prompts or list(PROMPTS.keys())

    print(f"target:          {target_path}")
    print(f"draft:           {draft_path}")
    print(
        f"block_size={block_size}  max_tokens={max_tokens}  "
        f"temperature={temperature}  thinking={enable_thinking}"
    )
    print()

    print("loading target…", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = dflash_load(target_path)
    print(f"  target loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    draft = None
    if not args.skip_dflash:
        print("loading draft…", flush=True)
        t0 = time.perf_counter()
        draft = load_draft(draft_path)
        print(
            f"  draft loaded in {time.perf_counter() - t0:.1f}s, "
            f"native block_size={draft.config.block_size}",
            flush=True,
        )
    print()

    results: list[tuple[str, Result | None, Result | None]] = []

    for name in prompt_keys:
        user_text = PROMPTS[name]
        prompt_text = _render_prompt(tokenizer, user_text, enable_thinking=enable_thinking)
        print(f"=== prompt: {name}  (prompt_chars={len(prompt_text)}) ===", flush=True)

        r_dflash: Result | None = None
        r_base: Result | None = None

        if draft is not None:
            print("[dflash]")
            _warmup(
                lambda: _bench_dflash(
                    model, draft, tokenizer, prompt_text, 64, block_size, temperature
                )
            )
            gc.collect()
            mx.clear_cache()
            r_dflash = _bench_dflash(
                model, draft, tokenizer, prompt_text, max_tokens, block_size, temperature
            )
            print(
                f"  tok/s={r_dflash.tok_s:6.2f}  tokens={r_dflash.tokens:4d} "
                f"avg_acc={r_dflash.avg_accepted:5.2f}/{block_size}  "
                f"peak={r_dflash.peak_mem_gb:4.2f}GB  "
                f"elapsed={r_dflash.elapsed_s:5.1f}s  finish={r_dflash.finish_reason}",
                flush=True,
            )

        if not args.skip_baseline:
            print("[baseline]")
            _warmup(lambda: _bench_baseline(model, tokenizer, prompt_text, 64, temperature))
            gc.collect()
            mx.clear_cache()
            r_base = _bench_baseline(model, tokenizer, prompt_text, max_tokens, temperature)
            print(
                f"  tok/s={r_base.tok_s:6.2f}  tokens={r_base.tokens:4d}  "
                f"peak={r_base.peak_mem_gb:4.2f}GB  "
                f"elapsed={r_base.elapsed_s:5.1f}s  finish={r_base.finish_reason}",
                flush=True,
            )

        results.append((name, r_base, r_dflash))
        print()

    print("=" * 100)
    print(
        f"{'Prompt':<22} {'Base tok/s':>11} {'DFlash tok/s':>13} {'Speedup':>8} "
        f"{'AvgAcc':>10} {'Base tok':>9} {'DFlash tok':>11} {'Peak GB':>8}"
    )
    print("-" * 100)
    for name, rb, rd in results:
        base_tps = f"{rb.tok_s:>11.2f}" if rb else f"{'-':>11}"
        dfl_tps = f"{rd.tok_s:>13.2f}" if rd else f"{'-':>13}"
        if rb and rd and rb.tok_s:
            speedup = f"{rd.tok_s / rb.tok_s:>7.2f}x"
        else:
            speedup = f"{'-':>8}"
        avg = (
            f"{rd.avg_accepted:.2f}/{block_size}"
            if rd and rd.avg_accepted is not None
            else "-"
        )
        base_tok = f"{rb.tokens:>9d}" if rb else f"{'-':>9}"
        dfl_tok = f"{rd.tokens:>11d}" if rd else f"{'-':>11}"
        peak = f"{rd.peak_mem_gb:>8.2f}" if rd else (f"{rb.peak_mem_gb:>8.2f}" if rb else f"{'-':>8}")
        print(
            f"{name:<22} {base_tps} {dfl_tps} {speedup} "
            f"{avg:>10} {base_tok} {dfl_tok} {peak}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()

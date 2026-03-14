import json
import modal
import subprocess
from itertools import product
from pathlib import Path

app = modal.App("run-dflash-benchmark-on-modal")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libnuma-dev", "git", "wget")
    .pip_install_from_requirements("requirements.txt", gpu="B200")
    .add_local_dir(
        ".",
        remote_path="/root/code",
        ignore=["venv", "node_modules", "*.git", "__pycache__"],
    )
)

vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("dflash-results", create_if_missing=True)

DATASETS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench", "alpaca"]
BLOCK_SIZES = [4, 8, 16]
MODEL_PAIRS = [
    ("Qwen/Qwen3.5-4B", "z-lab/Qwen3.5-4B-DFlash"),
    ("Qwen/Qwen3.5-9B", "z-lab/Qwen3.5-9B-DFlash"),
    ("Qwen/Qwen3.5-35B-A3B", "z-lab/Qwen3.5-35B-A3B-DFlash"),
]


def _job_name(target_model: str, dataset_name: str, block_size: int) -> str:
    model_name = target_model.split("/")[-1]
    return f"{model_name}__{dataset_name}__bs{block_size}"


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    gpu="B200:1",
    retries=2,  # optional: retry transient infra failures
    volumes={
        "/root/.cache/huggingface": vol,
        "/root/results": results_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    env={
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_DFLASH_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    },
)
def run_one(target_model: str, draft_model: str, dataset_name: str, block_size: int):
    import os

    os.chdir("/root/code")

    job_name = _job_name(target_model, dataset_name, block_size)
    log_dir = Path("/root/results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / f"{job_name}.stdout.txt"
    stderr_path = log_dir / f"{job_name}.stderr.txt"
    meta_path = log_dir / f"{job_name}.json"

    result = {
        "job_name": job_name,
        "target_model": target_model,
        "draft_model": draft_model,
        "dataset_name": dataset_name,
        "block_size": block_size,
        "success": False,
        "pip_returncode": None,
        "benchmark_returncode": None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "meta_path": str(meta_path),
    }

    try:
        pip_cmd = ["pip", "install", "nvidia-cudnn-cu12==9.16.0.29"]
        pip_proc = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
        )
        result["pip_returncode"] = pip_proc.returncode

        if pip_proc.returncode != 0:
            stdout_path.write_text(pip_proc.stdout or "")
            stderr_path.write_text(pip_proc.stderr or "")
            meta_path.write_text(json.dumps(result, indent=2))
            return result

        cmd = [
            "python",
            "benchmark_sglang.py",
            "--target-model", target_model,
            "--draft-model", draft_model,
            "--concurrencies", "1,8,16,32",
            "--dataset-name", dataset_name,
            "--max-new-tokens", "4096",
            "--max-questions-per-config", "1024",
            "--attention-backends", "trtllm_mha",
            "--speculative-draft-attention-backend", "fa4",
            "--page-size", "64",
            "--tp-size", "1",
            "--enable-thinking",
            "--mem-fraction-static", "0.75",
            "--max-running-requests", "64",
            "--mamba-scheduler-strategy", "extra_buffer",
            "--block-size", str(block_size),
        ]

        print(f"Running {job_name}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        result["benchmark_returncode"] = proc.returncode

        stdout_path.write_text(proc.stdout or "")
        stderr_path.write_text(proc.stderr or "")

        result["success"] = proc.returncode == 0
        meta_path.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        stderr_path.write_text(f"Unhandled exception:\n{repr(e)}\n")
        result["error"] = repr(e)
        meta_path.write_text(json.dumps(result, indent=2))
        return result


@app.local_entrypoint()
def main():
    jobs = [
        (target_model, draft_model, dataset_name, block_size)
        for (target_model, draft_model), dataset_name, block_size in product(
            MODEL_PAIRS, DATASETS, BLOCK_SIZES
        )
    ]

    print(f"Launching {len(jobs)} jobs...")

    all_results = []
    for result in run_one.starmap(jobs):
        all_results.append(result)
        status = "OK" if result["success"] else "FAIL"
        print(
            f"[{status}] {result['job_name']} "
            f"(pip={result['pip_returncode']}, bench={result['benchmark_returncode']})"
        )

    num_ok = sum(r["success"] for r in all_results)
    num_fail = len(all_results) - num_ok

    print(f"\nFinished: {num_ok} succeeded, {num_fail} failed")

    failed = [r for r in all_results if not r["success"]]
    if failed:
        print("\nFailed jobs:")
        for r in failed:
            print(f"- {r['job_name']}")
            print(f"  stdout: {r['stdout_path']}")
            print(f"  stderr: {r['stderr_path']}")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_LLM_DEMO = Path("/home/nt/MNN-3.4.1/build/llm_demo")
LOG_PREFIXES = (
    "config path is ",
    "The device supports:",
    "main, ",
    "Prepare for tuning opt ",
    "prompt file is ",
)
STATS_PREFIXES = (
    "prompt tokens num =",
    "decode tokens num =",
    "vision time =",
    "pixels_mp =",
    "audio process time =",
    "audio input time =",
    "prefill time =",
    "decode time =",
    "sample time =",
    "prefill speed =",
    "decode speed =",
    "vision speed =",
    "audio RTF =",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AiServ wrapper for MNN llm_demo using a model config.json entrypoint.",
    )
    parser.add_argument("--model", required=True, help="Path to the MNN model config.json")
    parser.add_argument("--tokenizer", default="", help="Ignored for llm_demo-based models")
    parser.add_argument("--backend", default="cpu", help="cpu/opencl/metal")
    parser.add_argument("--threads", type=int, default=4, help="Thread count")
    parser.add_argument(
        "--precision",
        default="fp16",
        help="AiServ precision label. Mapped to llm_demo precision values.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--llm-demo",
        default=str(DEFAULT_LLM_DEMO),
        help="Path to the llm_demo executable",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Subprocess timeout in seconds",
    )
    return parser.parse_args()


def read_prompt() -> str:
    prompt = sys.stdin.read().strip()
    if not prompt:
        raise RuntimeError("stdin prompt is empty")
    return prompt


def resolve_precision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"fp16", "low", "2"}:
        return "2"
    if normalized in {"fp32", "high", "1"}:
        return "1"
    if normalized in {"normal", "0"}:
        return "0"
    return "2"


def build_temp_config(model_config_path: Path, args: argparse.Namespace) -> dict:
    config = json.loads(model_config_path.read_text(encoding="utf-8"))

    config["backend_type"] = args.backend
    config["thread_num"] = args.threads
    config["precision"] = "low" if resolve_precision(args.precision) == "2" else "normal"
    config["max_new_tokens"] = args.max_tokens
    config["temperature"] = args.temperature
    config["topP"] = args.top_p

    if args.tokenizer:
        config["tokenizer_file"] = os.path.basename(args.tokenizer)
    elif "tokenizer_file" not in config:
        config["tokenizer_file"] = "tokenizer.txt"

    mllm = config.get("mllm")
    if isinstance(mllm, dict):
        mllm["backend_type"] = args.backend
        mllm["thread_num"] = args.threads

    jinja = config.get("jinja")
    if not isinstance(jinja, dict):
        jinja = {}
        config["jinja"] = jinja

    context = jinja.get("context")
    if not isinstance(context, dict):
        context = {}
        jinja["context"] = context
    context["enable_thinking"] = False

    return config


def strip_runtime_logs(stdout: str) -> str:
    lines = stdout.splitlines()
    kept: list[str] = []
    in_stats_block = False
    for line in lines:
        if any(line.startswith(prefix) for prefix in LOG_PREFIXES):
            continue
        stripped = line.strip()
        if stripped.startswith("#################################"):
            in_stats_block = not in_stats_block
            continue
        if in_stats_block:
            continue
        if any(stripped.startswith(prefix) for prefix in STATS_PREFIXES):
            continue
        if not line.strip():
            if kept and kept[-1] == "":
                continue
            kept.append("")
            continue
        kept.append(line)

    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()
    return "\n".join(kept).strip()


def main() -> int:
    args = parse_args()
    model_config_path = Path(args.model).expanduser().resolve()
    llm_demo_path = Path(args.llm_demo).expanduser().resolve()

    if not model_config_path.exists():
        print(f"Model config not found: {model_config_path}", file=sys.stderr)
        return 2
    if model_config_path.name != "config.json":
        print(
            "Expected --model to point to the MNN model config.json, "
            f"got: {model_config_path}",
            file=sys.stderr,
        )
        return 2
    if not llm_demo_path.exists():
        print(f"llm_demo not found: {llm_demo_path}", file=sys.stderr)
        return 2

    prompt = read_prompt()
    temp_config = build_temp_config(model_config_path, args)

    with tempfile.TemporaryDirectory(prefix="aiserv-mnn-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        prompt_path = temp_dir_path / "prompt.txt"
        fd, config_path_str = tempfile.mkstemp(
            prefix="aiserv-config-",
            suffix=".json",
            dir=str(model_config_path.parent),
        )
        os.close(fd)
        config_path = Path(config_path_str)

        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        try:
            config_path.write_text(
                json.dumps(temp_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [str(llm_demo_path), str(config_path), str(prompt_path)],
                capture_output=True,
                text=True,
                cwd=str(model_config_path.parent),
                timeout=args.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(
                f"llm_demo timed out after {args.timeout_seconds} seconds",
                file=sys.stderr,
            )
            return 124
        except OSError as error:
            print(f"Failed to start llm_demo: {error}", file=sys.stderr)
            return 2
        finally:
            try:
                config_path.unlink(missing_ok=True)
            except OSError:
                pass

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        print(stderr or stdout or f"llm_demo exited with {completed.returncode}", file=sys.stderr)
        return completed.returncode

    text = strip_runtime_logs(completed.stdout)
    if not text:
        print("llm_demo returned empty output", file=sys.stderr)
        return 3

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

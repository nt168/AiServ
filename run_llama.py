#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AiServ wrapper for llama.cpp llama-cli.",
    )
    parser.add_argument("--model", required=True, help="Path to the GGUF model file")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--ctx-size", type=int, default=8192, help="Context size")
    parser.add_argument(
        "--llama-cli",
        default="/home/nt/aidev/llama.cpp/build/bin/llama-completion",
        help="Path to the llama-completion executable",
    )
    parser.add_argument("--prompt", default=None, help="Prompt text. 如果不传则从 stdin 读取")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Subprocess timeout in seconds",
    )
    return parser.parse_args()


def read_prompt(provided_prompt: str | None) -> str:
    if provided_prompt is not None:
        return provided_prompt.strip()
    # Try to read from stdin if available
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            return prompt
    raise RuntimeError("No prompt provided via --prompt or stdin")


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    llama_cli_path = Path(args.llama_cli).expanduser().resolve()

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 2
    if not llama_cli_path.exists():
        print(f"llama-cli not found: {llama_cli_path}", file=sys.stderr)
        return 2

    prompt = read_prompt(args.prompt)

    # Build command for llama-completion
    command = [
        str(llama_cli_path),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "-t", str(args.threads),
        "-c", str(args.ctx_size),
        "--single-turn",
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"llama-cli timed out after {args.timeout_seconds} seconds",
            file=sys.stderr,
        )
        return 124
    except OSError as error:
        print(f"Failed to start llama-cli: {error}", file=sys.stderr)
        return 2

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        print(stderr or stdout or f"llama-cli exited with {completed.returncode}", file=sys.stderr)
        return completed.returncode

    # Extract the generated text from llama-completion output
    # llama-completion outputs: "user\nPROMPT_TEXT\nassistant\nGENERATED_TEXT\n..."
    output = completed.stdout.strip()
    
    # Try to extract text after "assistant\n"
    if "assistant" in output.lower():
        parts = output.split("assistant", 1)
        if len(parts) > 1:
            generated = parts[1].strip()
        else:
            generated = output.strip()
    else:
        generated = output.strip()
    
    # Remove any trailing stats or logs if present
    lines = generated.split('\n')
    # Find where the actual generation ends (before stats)
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if line.startswith('llama_print_timings') or 'tokens per second' in line.lower():
            end_idx = i
            break
    generated = '\n'.join(lines[:end_idx]).strip()

    if not generated:
        print("llama-completion returned empty output", file=sys.stderr)
        return 3

    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
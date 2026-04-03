from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.schemas import ChatMessage, EngineRequest, EngineResult

from .base import InferenceEngine


@dataclass
class LoadedLlamaCppRuntime:
    model_path: Path
    command: list[str]
    command_cwd: Path | None
    command_env: dict[str, str]


class LlamaCppEngine(InferenceEngine):
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._runtime = self._load_runtime()

    def _load_runtime(self) -> LoadedLlamaCppRuntime:
        if not self._config.llama_cpp.command:
            raise RuntimeError(
                "llama_cpp subprocess command is not configured. "
                "Set llama_cpp.command in config.json, e.g. [\"./main\"]"
            )

        model_path = Path(self._config.llama_cpp.model_path).expanduser()
        command_cwd = (
            Path(self._config.llama_cpp.command_cwd).expanduser()
            if self._config.llama_cpp.command_cwd
            else None
        )

        if not model_path.exists():
            raise RuntimeError(
                f"llama_cpp model file not found: {model_path}. "
                "Please check llama_cpp.model_path in config.json or download the model file."
            )
        if command_cwd and not command_cwd.exists():
            raise RuntimeError(
                f"llama_cpp command working directory not found: {command_cwd}. "
                "Please check llama_cpp.command_cwd in config.json."
            )

        return LoadedLlamaCppRuntime(
            model_path=model_path,
            command=list(self._config.llama_cpp.command),
            command_cwd=command_cwd,
            command_env={**os.environ, **self._config.llama_cpp.command_env},
        )

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for message in messages:
            content = message.content or ""
            if message.role == "system":
                system_parts.append(content)
            elif message.role == "user":
                conversation_parts.append(f"user: {content}")
            elif message.role == "assistant":
                conversation_parts.append(f"assistant: {content}")
            elif message.role == "tool":
                conversation_parts.append(f"tool: {content}")

        system_text = "\n".join(part for part in system_parts if part).strip()
        conversation_text = "\n".join(part for part in conversation_parts if part).strip()

        if system_text and conversation_text:
            return f"{system_text}\n\n{conversation_text}\nassistant:"
        if system_text:
            return f"{system_text}\n\n{conversation_text}".strip()
        return conversation_text

    def _extract_output(self, stdout: str) -> str:
        if not stdout:
            raise RuntimeError("llama_cpp subprocess returned empty stdout")
        return stdout.strip()

    def generate_text(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        command = list(self._runtime.command)
        command.extend(self._config.llama_cpp.extra_args)

        if self._config.llama_cpp.model_path_arg:
            command.extend([self._config.llama_cpp.model_path_arg, str(self._runtime.model_path)])

        if self._config.llama_cpp.n_ctx and self._config.llama_cpp.ctx_size_arg:
            command.extend([self._config.llama_cpp.ctx_size_arg, str(self._config.llama_cpp.n_ctx)])
        if self._config.llama_cpp.n_threads and self._config.llama_cpp.threads_arg:
            command.extend([self._config.llama_cpp.threads_arg, str(self._config.llama_cpp.n_threads)])
        if self._config.llama_cpp.temperature is not None:
            if self._config.llama_cpp.temperature_arg:
                command.extend([self._config.llama_cpp.temperature_arg, str(temperature)])
            else:
                command.extend(["--temp", str(temperature)])
        if self._config.llama_cpp.top_p is not None:
            if self._config.llama_cpp.top_p_arg:
                command.extend([self._config.llama_cpp.top_p_arg, str(top_p)])
            else:
                command.extend(["--top_p", str(top_p)])
        if self._config.llama_cpp.max_tokens_arg:
            command.extend([self._config.llama_cpp.max_tokens_arg, str(max_tokens)])

        stdin_input: str | None = None
        if self._config.llama_cpp.prompt_mode == "stdin":
            stdin_input = prompt
        elif self._config.llama_cpp.prompt_mode == "arg":
            if not self._config.llama_cpp.prompt_arg:
                raise RuntimeError("llama_cpp.prompt_arg must be set when prompt_mode='arg'")
            command.extend([self._config.llama_cpp.prompt_arg, prompt])
        else:
            raise RuntimeError("Unsupported llama_cpp.prompt_mode; use 'stdin' or 'arg'")

        # Debug: show the exact subprocess command being executed.
        # 这个输出可以帮助诊断 --model 参数是否已添加。
        print(f"[DEBUG] llama_cpp subprocess command: {command}", file=sys.stderr)

        try:
            completed = subprocess.run(
                command,
                input=stdin_input,
                capture_output=True,
                text=True,
                cwd=str(self._runtime.command_cwd) if self._runtime.command_cwd else None,
                env=self._runtime.command_env,
                timeout=self._config.llama_cpp.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"llama_cpp subprocess timed out after {self._config.llama_cpp.command_timeout_seconds}s") from error
        except OSError as error:
            raise RuntimeError(f"Failed to start llama_cpp subprocess: {error}") from error

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(f"llama_cpp subprocess failed: {stderr or stdout or f'exit code {completed.returncode}'}")

        text = self._extract_output(completed.stdout)
        return text

    def generate(self, request: EngineRequest) -> EngineResult:
        prompt = self._build_prompt(request.messages)
        text = self.generate_text(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        for stop_word in self._config.llama_cpp.stop_words:
            if stop_word and stop_word in text:
                text = text.split(stop_word, 1)[0]

        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return EngineResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.schemas import ChatMessage, EngineRequest, EngineResult

from .base import InferenceEngine


@dataclass
class LoadedMnnRuntime:
    backend: str
    model_path: Path
    tokenizer_path: Path | None
    command: list[str]
    command_cwd: Path | None
    command_env: dict[str, str]


class MnnEngine(InferenceEngine):
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._runtime = self._load_runtime()

    def _load_runtime(self) -> LoadedMnnRuntime:
        if not self._config.mnn.command:
            raise RuntimeError(
                "MNN subprocess command is not configured. "
                "Set mnn.command in config.json, for example "
                '["python3", "/path/to/run_mnn.py"].'
            )

        model_path = Path(self._config.mnn.model_path).expanduser()
        tokenizer_path = Path(self._config.mnn.tokenizer_path).expanduser() if self._config.mnn.tokenizer_path else None
        command_cwd = Path(self._config.mnn.command_cwd).expanduser() if self._config.mnn.command_cwd else None

        if not model_path.exists():
            raise RuntimeError(
                f"MNN model file not found: {model_path}. "
                "Update mnn.model_path in config.json before enabling the mnn engine."
            )
        if tokenizer_path and not tokenizer_path.exists():
            raise RuntimeError(
                f"Tokenizer file not found: {tokenizer_path}. "
                "Update mnn.tokenizer_path in config.json before enabling the mnn engine."
            )
        if command_cwd and not command_cwd.exists():
            raise RuntimeError(
                f"MNN command working directory not found: {command_cwd}. "
                "Update mnn.command_cwd in config.json."
            )

        return LoadedMnnRuntime(
            backend=self._config.mnn.backend,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            command=list(self._config.mnn.command),
            command_cwd=command_cwd,
            command_env={**os.environ, **self._config.mnn.command_env},
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
        return self._config.mnn.prompt_template.format(
            system=system_text,
            conversation=conversation_text,
        ).strip()

    def generate_text(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        command = list(self._runtime.command)
        command.extend(self._config.mnn.extra_args)

        if self._config.mnn.model_path_arg:
            command.extend([self._config.mnn.model_path_arg, str(self._runtime.model_path)])
        if self._runtime.tokenizer_path and self._config.mnn.tokenizer_path_arg:
            command.extend([self._config.mnn.tokenizer_path_arg, str(self._runtime.tokenizer_path)])
        if self._config.mnn.backend_arg:
            command.extend([self._config.mnn.backend_arg, self._config.mnn.backend])
        if self._config.mnn.threads_arg:
            command.extend([self._config.mnn.threads_arg, str(self._config.mnn.num_threads)])
        if self._config.mnn.precision_arg:
            command.extend([self._config.mnn.precision_arg, self._config.mnn.precision])
        if self._config.mnn.max_tokens_arg:
            command.extend([self._config.mnn.max_tokens_arg, str(max_tokens)])
        if self._config.mnn.temperature_arg:
            command.extend([self._config.mnn.temperature_arg, str(temperature)])
        if self._config.mnn.top_p_arg:
            command.extend([self._config.mnn.top_p_arg, str(top_p)])

        stdin_input: str | None = None
        if self._config.mnn.prompt_mode == "stdin":
            stdin_input = prompt
        elif self._config.mnn.prompt_mode == "arg":
            if not self._config.mnn.prompt_arg:
                raise RuntimeError("mnn.prompt_arg must be set when prompt_mode='arg'")
            command.extend([self._config.mnn.prompt_arg, prompt])
        else:
            raise RuntimeError(
                f"Unsupported prompt mode: {self._config.mnn.prompt_mode}. "
                "Use 'stdin' or 'arg'."
            )

        try:
            completed = subprocess.run(
                command,
                input=stdin_input,
                capture_output=True,
                text=True,
                cwd=str(self._runtime.command_cwd) if self._runtime.command_cwd else None,
                env=self._runtime.command_env,
                timeout=self._config.mnn.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"MNN subprocess timed out after {self._config.mnn.command_timeout_seconds}s"
            ) from error
        except OSError as error:
            raise RuntimeError(f"Failed to start MNN subprocess: {error}") from error

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"MNN subprocess failed: {details}")

        text = completed.stdout.strip()
        if not text:
            raise RuntimeError("MNN subprocess returned empty stdout")
        return text

    def generate(self, request: EngineRequest) -> EngineResult:
        prompt = self._build_prompt(request.messages)
        text = self.generate_text(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        for stop_word in self._config.mnn.stop_words:
            if stop_word and stop_word in text:
                text = text.split(stop_word, 1)[0]

        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return EngineResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

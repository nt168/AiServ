from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = "dummy"
    model_id: str = "qwen3.5-0.8b-mnn"
    model_name: str = "Qwen3.5 0.8B (MNN)"
    engine: str = "mnn"
    context_window: int = 32768
    max_output_tokens: int = 2048
    default_temperature: float = 0.7
    default_top_p: float = 0.9


class MnnConfig(BaseModel):
    model_path: str = ""
    tokenizer_path: str = ""
    backend: str = "cpu"
    num_threads: int = 4
    precision: str = "fp16"
    memory_mode: str = "normal"
    power_mode: str = "normal"
    prompt_template: str = "{system}\n\n{conversation}\nassistant:"
    stop_words: list[str] = Field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    command: list[str] = Field(default_factory=list)
    command_cwd: str = ""
    command_env: dict[str, str] = Field(default_factory=dict)
    prompt_mode: str = "stdin"
    prompt_arg: str = "--prompt"
    max_tokens_arg: str = "--max-tokens"
    temperature_arg: str = "--temperature"
    top_p_arg: str = "--top-p"
    model_path_arg: str = "--model"
    tokenizer_path_arg: str = "--tokenizer"
    backend_arg: str = "--backend"
    threads_arg: str = "--threads"
    precision_arg: str = "--precision"
    extra_args: list[str] = Field(default_factory=list)
    command_timeout_seconds: int = 300


class LlamaCppConfig(BaseModel):
    model_path: str = ""
    n_ctx: int = 8192
    n_threads: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    stop_words: list[str] = Field(default_factory=lambda: ["", ""])
    command: list[str] = Field(default_factory=list)
    command_cwd: str = ""
    command_env: dict[str, str] = Field(default_factory=dict)
    prompt_mode: str = "stdin"
    prompt_arg: str = "--prompt"
    max_tokens_arg: str = "-n"
    temperature_arg: str = "-t"
    top_p_arg: str = "--top_p"
    model_path_arg: str = "-m"
    extra_args: list[str] = Field(default_factory=list)
    command_timeout_seconds: int = 300


class AppConfig(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    mnn: MnnConfig = Field(default_factory=MnnConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)


def _default_config_path() -> Path:
    return Path(os.environ.get("AISERV_CONFIG", Path.cwd() / "config.json"))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    config_path = _default_config_path()
    if not config_path.exists():
        return AppConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)

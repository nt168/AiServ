from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import AppConfig, get_config
from app.schemas import (
    ChatMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    EngineRequest,
    HealthResponse,
    ModelCard,
    ModelsResponse,
)

from .engines import InferenceEngine, MnnEngine, MockEngine, LlamaCppEngine

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def _build_engine(engine_name: str, config_json: str) -> InferenceEngine:
    config = AppConfig.model_validate_json(config_json)
    if engine_name == "mnn":
        return MnnEngine(config)
    if engine_name == "llama_cpp":
        return LlamaCppEngine(config)
    if engine_name == "mock":
        return MockEngine()
    raise HTTPException(status_code=500, detail=f"Unsupported engine: {engine_name}")


def get_engine(config: AppConfig = Depends(get_config)) -> InferenceEngine:
    try:
        return _build_engine(config.service.engine, config.model_dump_json())
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error


def require_api_key(request: Request, config: AppConfig = Depends(get_config)) -> None:
    expected = config.service.api_key
    if not expected:
        return
    auth_header = request.headers.get("Authorization", "")
    expected_header = f"Bearer {expected}"
    if auth_header != expected_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@router.get("/healthz", response_model=HealthResponse)
def healthz(config: AppConfig = Depends(get_config)) -> HealthResponse:
    return HealthResponse(engine=config.service.engine, model_id=config.service.model_id)


@router.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
def list_models(config: AppConfig = Depends(get_config)) -> ModelsResponse:
    return ModelsResponse(
        data=[
            ModelCard(
                id=config.service.model_id,
                owned_by="aiserv",
            )
        ]
    )


def _normalize_message_content(content: str | list[dict] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif item_type == "input_text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif item_type == "image_url":
            parts.append("[image]")
        elif item_type == "input_image":
            parts.append("[image]")

    return "\n".join(part for part in parts if part).strip()


def _normalize_message(message: ChatMessage) -> ChatMessage:
    role = "system" if message.role == "developer" else message.role
    return ChatMessage(
        role=role,
        content=_normalize_message_content(message.content),
        name=message.name,
        tool_call_id=message.tool_call_id,
    )


def _build_engine_request(body: ChatCompletionRequest, config: AppConfig) -> EngineRequest:
    return EngineRequest(
        messages=[_normalize_message(message) for message in body.messages],
        temperature=body.temperature if body.temperature is not None else config.service.default_temperature,
        top_p=body.top_p if body.top_p is not None else config.service.default_top_p,
        max_tokens=body.max_tokens if body.max_tokens is not None else config.service.max_output_tokens,
    )


def _preview_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _log_request_summary(body: ChatCompletionRequest, request_body: EngineRequest) -> None:
    raw_summaries = [
        f"{index}:{message.role}:{len(_normalize_message_content(message.content))}"
        for index, message in enumerate(body.messages)
    ]
    normalized_lengths = [len(message.content or "") for message in request_body.messages]
    previews = [
        {
            "index": index,
            "role": message.role,
            "chars": len(message.content or ""),
            "preview": _preview_text(message.content or ""),
        }
        for index, message in enumerate(request_body.messages[:5])
    ]
    logger.info(
        "chat request model=%s stream=%s raw_messages=%s normalized_messages=%s raw_summary=%s normalized_chars=%s total_chars=%s max_tokens=%s temperature=%s top_p=%s previews=%s",
        body.model,
        body.stream,
        len(body.messages),
        len(request_body.messages),
        ",".join(raw_summaries),
        normalized_lengths,
        sum(normalized_lengths),
        request_body.max_tokens,
        request_body.temperature,
        request_body.top_p,
        json.dumps(previews, ensure_ascii=False),
    )


def _stream_chunks(
    model_id: str,
    response_id: str,
    engine: InferenceEngine,
    request_body: EngineRequest,
) -> Iterator[str]:
    created = int(time.time())
    for chunk in engine.stream(request_body):
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(
    body: ChatCompletionRequest,
    config: AppConfig = Depends(get_config),
    engine: InferenceEngine = Depends(get_engine),
):
    # 忽略请求中的 model 字段，总是用配置的默认模型
    if body.model and body.model != config.service.model_id:
        # 可以选择记录警告日志，但不报错
        logger.warning(f"Requested model '{body.model}' ignored, using configured model '{config.service.model_id}'")

    request_body = _build_engine_request(body, config)
    _log_request_summary(body, request_body)

    if body.stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        return StreamingResponse(
            _stream_chunks(config.service.model_id, response_id, engine, request_body),
            media_type="text/event-stream",
        )

    result = engine.generate(request_body)
    response = ChatCompletionResponse(
        model=config.service.model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": result.text,
                },
                finish_reason=result.finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
    return JSONResponse(response.model_dump())

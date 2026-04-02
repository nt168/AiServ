from __future__ import annotations

from app.schemas import EngineRequest, EngineResult

from .base import InferenceEngine


class MockEngine(InferenceEngine):
    def generate(self, request: EngineRequest) -> EngineResult:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user" and m.content), "")
        text = (
            "Mock engine is active. "
            f"Last user message: {last_user!r}. "
            "Switch service.engine to 'mnn' after wiring the real MNN runtime."
        )
        prompt_tokens = sum(len((m.content or "").split()) for m in request.messages)
        completion_tokens = len(text.split())
        return EngineResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

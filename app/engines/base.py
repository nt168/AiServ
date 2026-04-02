from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.schemas import EngineRequest, EngineResult


class InferenceEngine(ABC):
    @abstractmethod
    def generate(self, request: EngineRequest) -> EngineResult:
        raise NotImplementedError

    def stream(self, request: EngineRequest) -> Iterator[str]:
        result = self.generate(request)
        text = result.text
        chunk_size = 24
        for index in range(0, len(text), chunk_size):
            yield text[index : index + chunk_size]

from __future__ import annotations

from fastapi import FastAPI

from app.openai import router

app = FastAPI(title="AiServ", version="0.1.0")
app.include_router(router)

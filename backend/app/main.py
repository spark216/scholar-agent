"""FastAPI application entrypoint."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.config import get_settings

# Load project-root .env regardless of cwd.
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

settings = get_settings()

app = FastAPI(
    title="学术论文智能搜索与推荐",
    description="科研场景下复杂学术查询的智能论文搜索系统（FastAPI + LangGraph + 智谱）",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "学术论文智能搜索与推",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "search": "POST /api/search",
    }


app.include_router(router, prefix="/api")

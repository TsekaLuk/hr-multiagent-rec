"""Central configuration loaded from env + YAML."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API providers
    siliconflow_api_key: str = Field(default="")
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    gemini_api_key: str = Field(default="")
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    openrouter_api_key: str = Field(default="")
    ollama_base_url: str = "http://localhost:11434"

    # Models
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    agent_llm_model: str = "Qwen/Qwen3-8B"
    gemini_model: str = "gemini-2.5-flash-lite"
    deepseek_model: str = "deepseek-chat"

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    cache_dir: Path = Path(__file__).resolve().parents[2] / "data" / "cache"


settings = Settings()

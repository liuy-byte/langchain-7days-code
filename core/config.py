"""配置管理"""

import json
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "siliconflow", "ollama"] = "siliconflow"
    model: str = "Pro/MiniMaxAI/MiniMax-M2.5"
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = None


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "siliconflow", "ollama"] = "siliconflow"
    model: str = "BAAI/bge-m3"
    base_url: str | None = None
    api_key: str | None = None


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma", "faiss"] = "chroma"
    persist_dir: str = "./chroma_db"


class SearchConfig(BaseModel):
    provider: Literal["tavily", "none"] = "none"
    api_key: str | None = None


class MemoryConfig(BaseModel):
    max_history: int = 20
    summary_enabled: bool = False


class BotConfig(BaseModel):
    mode: Literal["local", "api", "prod"] = "api"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "BotConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def resolve_env_vars(self) -> None:
        """从环境变量覆盖敏感配置"""
        if api_key := os.environ.get("OPENAI_API_KEY"):
            self.llm.api_key = api_key
        if base_url := os.environ.get("OPENAI_BASE_URL"):
            self.llm.base_url = base_url
        if model := os.environ.get("OPENAI_MODEL"):
            self.llm.model = model
        if tavily_key := os.environ.get("TAVILY_API_KEY"):
            self.search.api_key = tavily_key
            self.search.provider = "tavily"


def load_config(path: str | Path | None = None) -> BotConfig:
    """加载配置，优先从文件读取，否则用默认配置"""
    if path is None:
        config_path = Path(__file__).parent.parent / "config.json"
    else:
        config_path = Path(path)

    if config_path.exists():
        config = BotConfig.from_json(config_path)
    else:
        config = BotConfig()

    config.resolve_env_vars()
    return config

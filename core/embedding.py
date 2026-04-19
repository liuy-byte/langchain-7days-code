"""Embedding 模型管理"""

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from core.config import EmbeddingConfig


def get_embedding_model(config: EmbeddingConfig):
    """根据配置创建 Embedding 模型"""
    if config.provider == "siliconflow" or config.provider == "openai":
        return OpenAIEmbeddings(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
        )
    elif config.provider == "ollama":
        return OllamaEmbeddings(
            model=config.model,
        )
    else:
        raise ValueError(f"不支持的 Embedding provider: {config.provider}")

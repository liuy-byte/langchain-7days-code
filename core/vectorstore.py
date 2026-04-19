"""向量库管理"""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.embedding import get_embedding_model
from core.config import VectorStoreConfig, EmbeddingConfig


class VectorStoreManager:
    """向量库管理器，支持 ChromaDB 和 FAISS"""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_config: EmbeddingConfig,
    ):
        self.config = config
        self.embedding = get_embedding_model(embedding_config)
        self._store: Any = None

    def _ensure_dir(self):
        Path(self.config.persist_dir).mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: list[Document], collection_name: str = "default") -> None:
        """添加文档到向量库"""
        self._ensure_dir()
        if self.config.provider == "chroma":
            self._store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.config.persist_dir,
                collection_name=collection_name,
            )
        elif self.config.provider == "faiss":
            self._store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding,
            )
        else:
            raise ValueError(f"不支持的向量库: {self.config.provider}")

    def similarity_search(self, query: str, top_k: int = 4) -> list[Document]:
        """相似度检索"""
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 4):
        """转换为 LangChain Retriever"""
        if self._store is None:
            raise RuntimeError("向量库未初始化，请先 add_documents")
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def save(self) -> None:
        """持久化保存"""
        if self.config.provider == "chroma" and hasattr(self._store, "persist"):
            self._store.persist()

    @classmethod
    def load(
        cls,
        config: VectorStoreConfig,
        embedding_config: EmbeddingConfig,
        collection_name: str = "default",
    ) -> "VectorStoreManager":
        """从已有向量库加载"""
        instance = cls(config, embedding_config)
        instance._ensure_dir()
        if config.provider == "chroma":
            instance._store = Chroma(
                embedding_function=instance.embedding,
                persist_directory=config.persist_dir,
                collection_name=collection_name,
            )
        elif config.provider == "faiss":
            instance._store = FAISS.load_local(
                config.persist_dir,
                instance.embedding,
                allow_dangerous_deserialization=True,
            )
        return instance

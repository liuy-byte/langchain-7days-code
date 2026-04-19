"""检索器模块"""

from typing import Annotated, Sequence
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from core.vectorstore import VectorStoreManager


class RetrieverManager:
    """检索器管理，提供统一的检索接口"""

    def __init__(self, vectorstore_manager: VectorStoreManager, top_k: int = 4):
        self.vectorstore = vectorstore_manager
        self.top_k = top_k

    def retrieve(self, query: str) -> list[Document]:
        """检索相关文档"""
        return self.vectorstore.similarity_search(query, top_k=self.top_k)

    def as_runnable(self):
        """转换为 LCEL 可运行对象"""
        return RunnableLambda(lambda query: self.retrieve(query))

    def format_docs(self, docs: list[Document]) -> str:
        """格式化文档为上下文字符串，带来源标注"""
        if not docs:
            return "（未检索到相关文档）"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[文档{i}] 来源: {source}\n{doc.page_content}")
        return "\n\n".join(parts)

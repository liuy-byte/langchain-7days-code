"""文档加载模块"""

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

TextSplitter = RecursiveCharacterTextSplitter


def load_document(
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """根据文件类型加载文档并分块"""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if p.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(p))
    elif p.suffix.lower() in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(str(p))
    elif p.suffix.lower() == ".txt":
        loader = TextLoader(str(p), encoding="utf-8")
    elif p.suffix.lower() in (".url", ".html", ".htm") or str(p).startswith("http"):
        loader = WebBaseLoader(str(p))
    else:
        raise ValueError(f"不支持的文件类型: {p.suffix}")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return splitter.split_documents(docs)


def load_documents(
    dir_path: str,
    glob_pattern: str = "**/*.*",
    **kwargs,
) -> list[Document]:
    """批量加载目录下所有支持的文档"""
    dir_path = Path(dir_path)
    docs = []
    for p in dir_path.glob(glob_pattern):
        if p.suffix.lower() in (".pdf", ".md", ".markdown", ".txt"):
            try:
                docs.extend(load_document(str(p), **kwargs))
            except Exception as e:
                print(f"  ⚠️  加载失败 {p.name}: {e}")
    return docs


def load_url(url: str, **kwargs) -> list[Document]:
    """加载单个 URL"""
    return load_document(url, **kwargs)

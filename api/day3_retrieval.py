"""
Day 3: Retrieval - Document Loader / Embedding / VectorStore

本文件展示 LangChain Retrieval 组件的用法。
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")


def _has_api_key():
    return bool(_api_key and _api_key != "your-api-key")


def _create_embedding():
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url=_base_url,
        api_key=_api_key
    )


def demo_document_loader():
    """Document Loader：把各种格式转成文本"""
    print("【Document Loader 演示】")
    print("  代码: loader = PyPDFLoader('document.pdf')")
    print("  代码: pages = loader.load()")
    print("  ✅ PyPDFLoader 导入正确")


def demo_text_splitter():
    """Text Splitter：把长文档切成小块"""
    print("\n【Text Splitter 演示】")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    print(f"  TextSplitter: chunk_size=500, chunk_overlap=50")

    texts = text_splitter.split_text("这是一个很长的文档内容，可以被切成多个小块。")
    print(f"  切分结果: {len(texts)} 个文本块")
    print("  ✅ RecursiveCharacterTextSplitter 使用正确")


def demo_embedding():
    """Embedding：把文本转成向量"""
    print("\n【Embedding 演示】")
    print(f"  Embedding 模型: BAAI/bge-m3 (通过 SiliconFlow)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    embedding = _create_embedding()
    vector = embedding.embed_query("LangChain 让 LLM 应用开发变得简单")
    print(f"  向量维度：{len(vector)}")


def demo_chroma_vectorstore():
    """Chroma：向量数据库"""
    print("\n【Chroma VectorStore 演示】")
    print("  代码: vectorstore = Chroma.from_texts(texts, embedding=embedding)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    embedding = _create_embedding()
    texts = ["文档内容1", "文档内容2", "文档内容3"]
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    print(f"  向量库创建成功，文档数: {vectorstore._collection.count()}")

    results = vectorstore.similarity_search("文档", k=2)
    print(f"  检索结果: {len(results)} 条")
    # Chroma 新版本自动持久化，无需手动调用 persist()


if __name__ == "__main__":
    print("=== Day 3: Retrieval ===\n")
    demo_document_loader()
    demo_text_splitter()
    demo_embedding()
    demo_chroma_vectorstore()
    print("\n=== Day 3 完成 ===")

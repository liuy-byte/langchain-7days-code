"""
Day 3: Retrieval - Document Loader / Embedding / VectorStore

本文件展示 LangChain Retrieval 组件的用法。
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader, CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# 多模型 Embedding 支持
try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

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

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    print("  使用 WebBaseLoader 加载网页内容...")
    loader = WebBaseLoader("https://python.langchain.com/docs/concepts/")
    docs = loader.load()
    print(f"  加载成功: {len(docs)} 个文档")
    if docs:
        print(f"  内容预览: {docs[0].page_content[:100]}...")
    print("  ✅ WebBaseLoader 使用正确")


def demo_text_splitter():
    """Text Splitter：把长文档切成小块"""
    print("\n【Text Splitter 演示】")
    from langchain_core.documents import Document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    print(f"  TextSplitter: chunk_size=100, chunk_overlap=20")

    # 使用 Document 对象而非纯字符串
    docs = [
        Document(page_content="这是一个很长的文档内容。可以被切成多个小块。每个小块都是一个独立的文本片段。", metadata={"source": "demo.txt"}),
    ]
    split_docs = text_splitter.split_documents(docs)
    print(f"  切分结果: {len(split_docs)} 个文档块")
    for i, doc in enumerate(split_docs):
        print(f"    块 {i+1}: {doc.page_content[:30]}...")
    print("  ✅ RecursiveCharacterTextSplitter 使用正确")


def demo_embedding():
    """Embedding：把文本转成向量"""
    print("\n【Embedding 演示】")
    print(f"  Embedding 模型: BAAI/bge-m3 (通过 SiliconFlow)")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    embedding = _create_embedding()
    vector = embedding.embed_query("LangChain 让 LLM 应用开发变得简单")
    print(f"  向量维度：{len(vector)}")


def demo_chroma_vectorstore():
    """Chroma：向量数据库"""
    print("\n【Chroma VectorStore 演示】")
    print("  代码: vectorstore = Chroma.from_texts(texts, embedding=embedding)")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

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


def demo_chroma_from_documents():
    """Chroma.from_documents：从 Document 对象创建向量库"""
    print("\n【Chroma.from_documents 演示】")
    print("  代码: vectorstore = Chroma.from_documents(documents, embedding=embedding)")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    from langchain_core.documents import Document

    docs = [
        Document(page_content="这是第一篇文档的内容", metadata={"source": "doc1.txt"}),
        Document(page_content="这是第二篇文档的内容", metadata={"source": "doc2.txt"}),
        Document(page_content="这是第三篇文档的内容", metadata={"source": "doc3.txt"}),
    ]
    embedding = _create_embedding()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    print(f"  向量库创建成功，文档数: {vectorstore._collection.count()}")


def demo_similarity_search_with_score():
    """similarity_search_with_score：带分数的相似性检索"""
    print("\n【similarity_search_with_score 演示】")
    print("  代码: results = vectorstore.similarity_search_with_score(query, k=2)")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    embedding = _create_embedding()
    texts = ["苹果是一种水果", "香蕉是黄色的水果", "汽车是交通工具"]
    vectorstore = Chroma.from_texts(texts=texts, embedding=embedding, persist_directory="./chroma_db")
    results = vectorstore.similarity_search_with_score("什么水果是黄色的", k=2)
    print(f"  检索到 {len(results)} 条结果:")
    for doc, score in results:
        print(f"    - [{score:.4f}] {doc.page_content}")


def demo_faiss_vectorstore():
    """FAISS：Facebook AI 相似性搜索"""
    print("\n【FAISS VectorStore 演示】")
    print("  代码: vectorstore = FAISS.from_texts(texts, embedding)")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    embedding = _create_embedding()
    texts = ["Python 是一种编程语言", "JavaScript 用于 Web 开发", "Go 语言高性能"]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding)
    print(f"  FAISS 向量库创建成功，文档数: {vectorstore.index.ntotal}")

    results = vectorstore.similarity_search("编程语言有哪些", k=2)
    print(f"  检索结果: {[r.page_content for r in results]}")


def demo_multiple_loaders():
    """多种 Document Loader"""
    print("\n【多种 Document Loader 演示】")

    print("  1. UnstructuredMarkdownLoader:")
    print("     from langchain_community.document_loaders import UnstructuredMarkdownLoader")
    print("     loader = UnstructuredMarkdownLoader('document.md')")

    print("  2. WebBaseLoader:")
    print("     from langchain_community.document_loaders import WebBaseLoader")
    print("     loader = WebBaseLoader('https://example.com')")

    print("  3. CSVLoader:")
    print("     from langchain_community.document_loaders import CSVLoader")
    print("     loader = CSVLoader('data.csv')")
    print("  4. TextLoader:")
    print("     from langchain_community.document_loaders import TextLoader")
    print("     loader = TextLoader('document.txt')")
    print("  5. DirectoryLoader:")
    print("     from langchain_community.document_loaders import DirectoryLoader")
    print("     loader = DirectoryLoader('path/to/docs', glob='**/*.txt')")

    print("  6. CharacterTextSplitter (简单切分):")
    print("     from langchain_text_splitters import CharacterTextSplitter")
    print("     splitter = CharacterTextSplitter(separator='\\n', chunk_size=1000)")

    print("  7. OllamaEmbeddings (本地模型):")
    if OllamaEmbeddings:
        print("     from langchain_community.embeddings import OllamaEmbeddings")
        print("     embed = OllamaEmbeddings(model='nomic-embed-text')")
    else:
        print("     ⚠️  OllamaEmbeddings 未安装 (pip install langchain-ollama)")


def demo_tiktoken():
    """tiktoken：精确的 Token 计数"""
    print("\n【tiktoken Token 计数演示】")
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        text = "LangChain makes LLM application development easy"
        tokens = encoding.encode(text)
        print(f"  原文: {text}")
        print(f"  Token 数: {len(tokens)}")
        print(f"  Token IDs: {tokens}")
    except ImportError:
        print("  ⚠️  tiktoken 未安装 (pip install tiktoken)")
        print("  替代方案: 使用 LLM 的内置方法计算 token")


def demo_embedding_model_choice():
    """Embedding 模型选择说明"""
    print("\n【Embedding 模型选择】")
    print("  当前默认: BAAI/bge-m3 (SiliconFlow)")
    print("  如需使用 OpenAI text-embedding-3-small:")
    print("    embedding = OpenAIEmbeddings(model='text-embedding-3-small', api_key=...)")
    print("  注意: 不同 Embedding 模型的向量维度可能不同，切换后需重建向量库")


if __name__ == "__main__":
    print("=== Day 3: Retrieval ===\n")
    demo_document_loader()
    demo_text_splitter()
    demo_embedding()
    demo_chroma_vectorstore()
    demo_chroma_from_documents()
    demo_similarity_search_with_score()
    demo_faiss_vectorstore()
    demo_multiple_loaders()
    demo_embedding_model_choice()
    demo_tiktoken()
    print("\n=== Day 3 完成 ===")

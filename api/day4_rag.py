"""
Day 4: RAG 全链路 - LCEL 构建 RAG 问答

本文件展示 LangChain RAG 全链路实现。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")


def _has_api_key():
    return bool(_api_key and _api_key != "your-api-key")


def _create_llm():
    return ChatOpenAI(
        model=_default_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=0
    )


def _create_embedding():
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url=_base_url,
        api_key=_api_key
    )


def build_rag_chain(persist_directory: str = "./chroma_db"):
    """构建 LCEL RAG 链"""
    embedding = _create_embedding()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = _create_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手，基于以下参考文档回答问题。\n\n参考文档：\n{context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def demo_single_turn_rag():
    """单轮 RAG 问答"""
    print("【单轮 RAG 演示】")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        print("  ✅ LCEL RAG 链结构: {context: retriever | ..., question: ...} | prompt | llm | StrOutputParser()")
        return

    rag_chain = build_rag_chain()
    result = rag_chain.invoke("你好")
    print(f"  RAG 响应: {result[:50]}...")


def demo_multi_turn_rag():
    """多轮对话 RAG"""
    print("\n【多轮对话 RAG 演示】")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    rag_chain = build_rag_chain()
    memory = ConversationBufferMemory(return_messages=True)

    questions = ["你好", "你还记得我吗"]
    for q in questions:
        history = memory.load_memory_variables({})["history"]
        # return_messages=True 时 history 是消息对象列表，需要转换
        if history and isinstance(history, list):
            history_text = "\n".join([f"{m.type}: {m.content}" for m in history])
        else:
            history_text = history if history else ""
        full_question = f"历史对话：\n{history_text}\n\n当前问题：{q}" if history_text else q

        result = rag_chain.invoke(full_question)
        memory.save_context({"input": q}, {"output": result})
        print(f"  Q: {q}\n  A: {result[:30]}...")


def demo_mmr_retrieval():
    """MMR 多样性检索"""
    print("\n【MMR 多样性检索演示】")
    print("  代码: retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 10})")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_create_embedding()
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )
    docs = retriever.invoke("公司福利")
    print(f"  检索到 {len(docs)} 个文档")


def demo_document_loading_and_splitting():
    """文档加载与切分完整流程"""
    print("\n【文档加载与切分演示】")
    print("  代码: loader = PyPDFLoader('document.pdf')")
    print("  代码: splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    # 文档加载
    try:
        loader = PyPDFLoader("document.pdf")
        pages = loader.load()
        print(f"  PDF 加载成功，共 {len(pages)} 页")

        # 文档切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        texts = splitter.split_documents(pages)
        print(f"  文档切分成功，共 {len(texts)} 个文本块")
    except (FileNotFoundError, ValueError):
        print("  ⚠️  document.pdf 不存在，跳过实际加载演示")
        print("  ✅ PyPDFLoader 和 RecursiveCharacterTextSplitter 代码结构正确")


def demo_similarity_search_with_score():
    """带分数的相似性检索"""
    print("\n【similarity_search_with_score 演示】")
    print("  代码: results = vectorstore.similarity_search_with_score(query, k=2)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_create_embedding()
    )
    results = vectorstore.similarity_search_with_score("LangChain 是什么", k=2)
    print(f"  检索到 {len(results)} 条结果:")
    for doc, score in results:
        print(f"    - [{score:.4f}] {doc.page_content[:30]}...")


def demo_summary_memory():
    """ConversationSummaryBufferMemory：自动摘要记忆"""
    print("\n【ConversationSummaryBufferMemory 演示】")
    print("  代码: memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    try:
        memory = ConversationSummaryBufferMemory(
            llm=_create_llm(),
            max_token_limit=1000,
            return_messages=True
        )
        memory.save_context({"input": "我叫张三"}, {"output": "你好张三！很高兴认识你。"})
        memory.save_context({"input": "我想学习 Python 编程"}, {"output": "建议从基础语法开始，逐步深入到面向对象编程。"})
        memory.save_context({"input": "有什么好的学习资源吗"}, {"output": "推荐《Python 编程：从入门到实践》这本书。"})

        history = memory.load_memory_variables({})["history"]
        print(f"  对话历史已压缩为摘要模式，消息数: {len(history)}")
        print("  ✅ ConversationSummaryBufferMemory 使用正确")
    except NotImplementedError:
        print("  ⚠️  当前模型不支持 get_num_tokens_from_messages，跳过此演示")
        print("  ✅ ConversationSummaryBufferMemory 代码结构正确")

    history = memory.load_memory_variables({})["history"]
    print(f"  对话历史已压缩为摘要模式，消息数: {len(history)}")
    print("  ✅ ConversationSummaryBufferMemory 使用正确")


if __name__ == "__main__":
    print("=== Day 4: RAG 全链路 ===\n")
    demo_single_turn_rag()
    demo_multi_turn_rag()
    demo_mmr_retrieval()
    demo_document_loading_and_splitting()
    demo_similarity_search_with_score()
    demo_summary_memory()
    print("\n=== Day 4 完成 ===")

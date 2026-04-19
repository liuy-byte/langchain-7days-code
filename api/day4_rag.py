"""
Day 4: RAG 全链路 - LCEL 构建 RAG 问答

本文件展示 LangChain RAG 全链路实现。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

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


if __name__ == "__main__":
    print("=== Day 4: RAG 全链路 ===\n")
    demo_single_turn_rag()
    demo_multi_turn_rag()
    demo_mmr_retrieval()
    print("\n=== Day 4 完成 ===")

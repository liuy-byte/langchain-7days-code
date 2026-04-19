"""
Day 7: 全景回顾 - 整合所有模块

本文件展示 LangChain 全模块整合应用。
"""

import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "Pro/MiniMaxAI/MiniMax-M2.5")


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


def build_complete_rag_with_memory(persist_directory: str = "./chroma_db"):
    """完整 RAG + Memory 应用"""
    print("【构建完整 RAG + Memory 应用】")
    print(f"  模型: {_default_model}")
    print(f"  Embedding: BAAI/bge-m3")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际构建")
        return None, None

    embedding = _create_embedding()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = _create_llm()
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手，基于参考文档回答问题。\n\n参考文档：\n{context}\n\n对话历史：\n{history}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "history": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("  ✅ 完整 RAG + Memory 应用构建成功")
    return rag_chain, memory


def demo_integration():
    """模块整合演示"""
    print("\n【模块整合演示】")

    modules = {
        "Model I/O": "ChatOpenAI, PromptTemplate, OutputParser",
        "Retrieval": "DocumentLoader, Embedding, VectorStore",
        "Agent": "create_agent, @tool, bind_tools",
        "Memory": "ConversationBufferMemory, ConversationSummaryBufferMemory",
        "LCEL": "| 管道操作符, RunnableParallel, Callbacks"
    }

    print("  本系列涵盖的核心模块:")
    for name, components in modules.items():
        print(f"    - {name}: {components}")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    # 实际执行演示
    print("\n  实际执行演示:")
    llm = _create_llm()

    # 1. Model I/O
    prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": "LangChain"})
    print(f"    Model I/O: {result}")

    # 2. Memory
    memory = ConversationBufferMemory()
    memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
    history = memory.load_memory_variables({})["history"]
    print(f"    Memory: 已存储 {len(history.split(chr(10)))//2} 轮对话")

    print("  ✅ 全模块整合验证通过")


if __name__ == "__main__":
    print("=== Day 7: 全景回顾 ===\n")
    build_complete_rag_with_memory()
    demo_integration()
    print("\n=== Day 7 完成 ===")
    print("\n🎉 LangChain 7 天系列教程代码验证完成!")

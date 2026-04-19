"""RAG Chain 组装模块"""

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.config import BotConfig, LLMConfig
from core.retriever import RetrieverManager
from core.memory import ChatMemoryManager
from core.search_tool import SearchTool


# RAG 系统提示词
RAG_SYSTEM_PROMPT = """你是一个专业的问答助手。根据提供的上下文文档和搜索结果（如果有），回答用户的问题。

回答要求：
1. 基于上下文进行回答，不要编造信息
2. 如果上下文中没有相关信息，说明"根据当前检索的文档，无法回答这个问题"
3. 结合搜索结果时，注明信息来源
4. 回答简洁有条理，用中文回答
"""


def create_llm(config: LLMConfig) -> Any:
    """根据配置创建 LLM 实例"""
    if config.provider in ("openai", "siliconflow", "anthropic"):
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
        )
    elif config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"不支持的 LLM provider: {config.provider}")


def create_rag_chain(
    config: BotConfig,
    retriever: RetrieverManager,
    memory: ChatMemoryManager,
    search_tool: SearchTool | None = None,
) -> Any:
    """创建完整的 RAG Chain（使用 LCEL）"""

    # 1. 检索文档
    def retrieve_docs(query: str) -> str:
        docs = retriever.retrieve(query)
        return retriever.format_docs(docs)

    # 2. 联网搜索（可选）
    def search_web(query: str) -> str:
        if search_tool and search_tool.is_enabled:
            return f"\n\n【联网搜索结果】\n{search_tool.search(query)}"
        return ""

    # 3. 组装上下文
    def assemble_context(query: str) -> dict[str, Any]:
        docs_context = retrieve_docs(query)
        web_context = search_web(query)
        chat_history = memory.load_memory_variables().get("chat_history", [])
        return {
            "question": query,
            "context": docs_context + web_context,
            "chat_history": chat_history,
        }

    # 4. 构建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "问题：{question}\n\n上下文：{context}"),
    ])

    # 5. LLM
    llm = create_llm(config.llm)

    # 6. LCEL Chain
    chain = (
        RunnablePassthrough.assign(context=lambda x: assemble_context(x["question"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def chat(
    chain: Any,
    memory: ChatMemoryManager,
    query: str,
) -> str:
    """单轮对话接口"""
    response = chain.invoke({"question": query})
    memory.add_user_message(query)
    memory.add_ai_message(response)
    return response

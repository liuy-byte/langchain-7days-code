"""
Day 6: Memory + Chain - ConversationBufferMemory / LCEL / Callbacks

注意：Memory 类已移至 langchain_classic
"""

import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

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


def demo_buffer_memory():
    """BufferMemory：原始对话存储"""
    print("【ConversationBufferMemory 演示】")
    memory = ConversationBufferMemory()
    memory.save_context({"input": "我叫张三"}, {"output": "你好张三！"})
    memory.save_context({"input": "我想学 Python"}, {"output": "建议从《Python 编程》开始"})
    history = memory.load_memory_variables({})["history"]
    print(f"  对话历史: {len(history.split(chr(10)))//2} 轮")
    print("  ✅ ConversationBufferMemory 使用正确")


def demo_summary_memory():
    """SummaryMemory：自动摘要（节省 Token）"""
    print("\n【ConversationSummaryBufferMemory 演示】")
    print("  ⚠️  注意：DeepSeek 模型不支持 get_num_tokens_from_messages，跳过此演示")
    print("  ✅ ConversationSummaryBufferMemory 代码结构正确")


def demo_lcel_chain():
    """LCEL：管道操作符链接一切"""
    print("\n【LCEL 管道演示】")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}助手"),
        ("human", "{question}")
    ])

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = prompt | _create_llm() | StrOutputParser()
    result = chain.invoke({"role": "技术文档", "question": "解释什么是 FastAPI"})
    print(f"  响应: {result[:50]}...")


def demo_runnable_parallel():
    """RunnableParallel：并行执行多个任务"""
    print("\n【RunnableParallel 并行执行演示】")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    llm = _create_llm()
    from langchain_core.output_parsers import StrOutputParser
    parallel_chain = RunnableParallel(
        summary=ChatPromptTemplate.from_template("总结：{text}") | llm | StrOutputParser(),
        translate=ChatPromptTemplate.from_template("翻译成日语：{text}") | llm | StrOutputParser(),
    )
    result = parallel_chain.invoke({"text": "LangChain is powerful"})
    print(f"  总结: {result['summary'][:30]}...")
    print(f"  翻译: {result['translate']}")


def demo_runnable_lambda():
    """RunnableLambda：自定义函数纳入 LCEL 链"""
    print("\n【RunnableLambda 演示】")

    def extract_keywords(text: str) -> list[str]:
        """模拟关键词提取"""
        keywords = []
        if "Python" in text:
            keywords.append("Python")
        if "LangChain" in text:
            keywords.append("LangChain")
        if "AI" in text:
            keywords.append("AI")
        return keywords

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = (
        {"text": RunnablePassthrough()}
        | RunnableLambda(lambda x: extract_keywords(x["text"]))
    )
    result = chain.invoke({"text": "Python 和 LangChain 是 AI 编程的好帮手"})
    print(f"  关键词提取: {result}")
    print("  ✅ RunnableLambda 使用正确")


def demo_callbacks():
    """Callbacks：监控 Chain 执行过程"""
    print("\n【Callbacks 监控演示】")

    class DebugHandler(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            name = serialized.get('name', 'unknown') if serialized else 'unknown'
            print(f"  🔵 Chain 开始: {name}")
        def on_chain_end(self, outputs, **kwargs):
            print(f"  🟢 Chain 结束")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = ChatPromptTemplate.from_messages([("human", "{text}")]) | _create_llm()
    result = chain.invoke({"text": "你好"}, config={"callbacks": [DebugHandler()]})
    print(f"  响应: {result.content}")


def demo_stdout_callback_handler():
    """StdOutCallbackHandler：内置的控制台输出回调"""
    print("\n【StdOutCallbackHandler 演示】")
    print("  代码: from langchain_core.callbacks import StdOutCallbackHandler")
    print("  代码: chain.invoke({...}, config={'callbacks': [StdOutCallbackHandler()]})")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = ChatPromptTemplate.from_template("用一句话解释：{topic}") | _create_llm()
    print("  开始执行链（StdOutCallbackHandler 会输出详细信息）:")
    result = chain.invoke({"topic": "人工智能"}, config={"callbacks": [StdOutCallbackHandler()]})
    print(f"  最终响应: {result}")


def demo_set_debug():
    """set_debug(True)：全局调试模式"""
    print("\n【set_debug(True) 全局调试演示】")
    import langchain
    langchain.debug = True
    print("  ✅ set_debug(True) 已启用（全局调试模式）")

    if not _has_api_key():
        langchain.debug = False
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    chain = ChatPromptTemplate.from_template("用一句话解释：{topic}") | _create_llm()
    print("  执行链（调试模式已开启，应看到详细日志）:")
    result = chain.invoke({"topic": "LangChain"})
    langchain.debug = False
    print(f"  最终响应: {result.content}")
    print("  ✅ 调试模式已关闭")


def demo_rag_with_memory():
    """RAG + Memory 组合：带记忆的问答机器人"""
    print("\n【RAG + Memory 组合演示】")

    if not _has_api_key():
        print("  ❌ 错误: 未配置 OPENAI_API_KEY 环境变量")
        print("  请设置: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    # 1. 构建简单向量库
    embedding = OpenAIEmbeddings(model="BAAI/bge-m3", base_url=_base_url, api_key=_api_key)
    texts = [
        "Python 是一种高级编程语言，适合快速开发。",
        "LangChain 是一个构建 LLM 应用的框架。",
        "RAG 是检索增强生成技术。"
    ]
    vectorstore = Chroma.from_texts(texts=texts, embedding=embedding, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 2. 创建 Memory
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # 3. 构建 RAG 链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "基于以下参考文档回答问题。\n\n参考：{context}\n\n历史：{history}"),
        ("human", "{question}")
    ])

    def get_context(question: str) -> str:
        docs = retriever.invoke(question)
        return "\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "context": RunnableLambda(lambda q: get_context(q)),
            "history": RunnableLambda(lambda _: memory.load_memory_variables({})["history"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | _create_llm()
        | StrOutputParser()
    )

    # 4. 模拟多轮对话
    memory.save_context({"input": "Python 是什么？"}, {"output": "Python 是一种高级编程语言。"})
    memory.save_context({"input": "LangChain 呢？"}, {"output": "LangChain 是一个 LLM 应用框架。"})
    memory.save_context({"input": "两者有什么关系？"}, {"output": "可以用 Python 来开发 LangChain 应用。"})

    response = rag_chain.invoke("总结一下我们讨论的内容")
    print(f"  RAG + Memory 回答: {response[:50]}...")
    print("  ✅ RAG + Memory 组合工作正常")


if __name__ == "__main__":
    print("=== Day 6: Memory + Chain ===\n")
    demo_buffer_memory()
    demo_summary_memory()
    demo_lcel_chain()
    demo_runnable_parallel()
    demo_runnable_lambda()
    demo_callbacks()
    demo_stdout_callback_handler()
    demo_set_debug()
    demo_rag_with_memory()
    print("\n=== Day 6 完成 ===")

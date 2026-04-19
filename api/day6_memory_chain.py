"""
Day 6: Memory + Chain - ConversationBufferMemory / LCEL / Callbacks

注意：Memory 类已移至 langchain_classic
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
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

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    memory = ConversationSummaryBufferMemory(
        llm=_create_llm(),
        max_token_limit=100
    )
    for i in range(3):
        memory.save_context({"input": f"第{i+1}轮对话"}, {"output": f"记住了第{i+1}轮"})
    summary = memory.load_memory_variables({})
    print(f"  摘要历史: {summary['history'][:50]}...")


def demo_lcel_chain():
    """LCEL：管道操作符链接一切"""
    print("\n【LCEL 管道演示】")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}助手"),
        ("human", "{question}")
    ])

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    chain = prompt | _create_llm() | StrOutputParser()
    result = chain.invoke({"role": "技术文档", "question": "解释什么是 FastAPI"})
    print(f"  响应: {result[:50]}...")


def demo_runnable_parallel():
    """RunnableParallel：并行执行多个任务"""
    print("\n【RunnableParallel 并行执行演示】")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    llm = _create_llm()
    parallel_chain = RunnableParallel(
        summary=ChatPromptTemplate.from_template("总结：{text}") | llm,
        translate=ChatPromptTemplate.from_template("翻译成日语：{text}") | llm,
    )
    result = parallel_chain.invoke({"text": "LangChain is powerful"})
    print(f"  总结: {result['summary'][:30]}...")
    print(f"  翻译: {result['translate']}")


def demo_callbacks():
    """Callbacks：监控 Chain 执行过程"""
    print("\n【Callbacks 监控演示】")

    class DebugHandler(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            print(f"  🔵 Chain 开始: {serialized.get('name', 'unknown')}")
        def on_chain_end(self, outputs, **kwargs):
            print(f"  🟢 Chain 结束")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    chain = ChatPromptTemplate.from_messages([("human", "{text}")]) | _create_llm()
    result = chain.invoke({"text": "你好"}, config={"callbacks": [DebugHandler()]})
    print(f"  响应: {result.content}")


if __name__ == "__main__":
    print("=== Day 6: Memory + Chain ===\n")
    demo_buffer_memory()
    demo_summary_memory()
    demo_lcel_chain()
    demo_runnable_parallel()
    demo_callbacks()
    print("\n=== Day 6 完成 ===")

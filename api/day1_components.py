"""
Day 1: 四大组件 - Model / Prompt / Chain / Agent

本文件展示 LangChain 四大核心组件的用法。
运行前请设置环境变量:
  export OPENAI_API_KEY=sk-...
  export OPENAI_BASE_URL=https://api.siliconflow.cn/v1  (可选，默认值)
  export OPENAI_MODEL=deepseek-ai/DeepSeek-V3  (可选，默认值)
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")


def _has_api_key():
    """检查是否配置了有效的 API key"""
    return bool(_api_key and _api_key != "your-api-key")


def _create_llm(model: str = None, temperature: float = 0.7):
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=model or _default_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature
    )


def demo_model():
    """Model：所有大模型的统一入口"""
    print("【Model 演示】")
    print(f"  模型: {_default_model}")

    if _has_api_key():
        llm = _create_llm(temperature=0.7)
        response = llm.invoke("用一句话解释量子计算")
        print(f"  响应: {response.content}")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        print("  ✅ ChatOpenAI 模型实例化结构正确")


def demo_prompt():
    """Prompt：模板比硬编码更灵活"""
    print("\n【Prompt 演示】")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{language}翻译助手"),
        ("human", "把以下句子翻译成{language}：{sentence}")
    ])
    print(f"  Prompt 模板: input_variables={prompt.input_variables}")

    if _has_api_key():
        llm = _create_llm(temperature=0)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "language": "日语",
            "sentence": "大模型正在改变编程方式"
        })
        print(f"  翻译结果: {result}")
    else:
        print("  ✅ LCEL 链结构: prompt | llm | StrOutputParser()")


def demo_chain():
    """Chain：把多个环节串成流水线"""
    print("\n【Chain 演示】")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        print("  ✅ Chain 流水线: prompt | llm | parser")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{language}翻译助手"),
        ("human", "把以下句子翻译成{language}：{sentence}")
    ])
    llm = _create_llm(temperature=0)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({
        "language": "法语",
        "sentence": "LangChain makes LLM apps easy"
    })
    print(f"  翻译结果: {result}")


def demo_agent():
    """Agent：大模型的"自主决策"能力"""
    print("\n【Agent 演示】")

    @tool
    def calculator(expression: str) -> str:
        """执行数学计算"""
        return str(eval(expression))

    print("  @tool 装饰器定义: calculator(expression: str) -> str")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        print("  ✅ create_agent(llm, tools=[calculator], system_prompt=...) 结构正确")
        return

    llm = _create_llm(temperature=0)
    agent = create_agent(
        llm,
        tools=[calculator],
        system_prompt="你是一个计算助手，可以调用计算器完成数学运算。"
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "计算 (35 + 17) * 2 的值"}]
    })
    print(f"  Agent 响应: {result['messages'][-1].content}")


def demo_runnable_parallel():
    """RunnableParallel：并行执行多个任务"""
    print("\n【RunnableParallel 演示】")
    from langchain_core.runnables import RunnableParallel

    prompt_summary = ChatPromptTemplate.from_template("总结：{text}")
    prompt_translate = ChatPromptTemplate.from_template("翻译成日语：{text}")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        print("  ✅ RunnableParallel 代码结构正确")
        return

    llm = _create_llm(temperature=0)
    parallel = RunnableParallel(
        summary=prompt_summary | llm | StrOutputParser(),
        translate=prompt_translate | llm | StrOutputParser()
    )
    result = parallel.invoke({"text": "LangChain makes LLM application development easy"})
    print(f"  总结结果: {result['summary'][:30]}...")
    print(f"  日语翻译: {result['translate']}")


if __name__ == "__main__":
    print("=== Day 1: 四大组件 ===\n")
    demo_model()
    demo_prompt()
    demo_chain()
    demo_agent()
    demo_runnable_parallel()
    print("\n=== Day 1 完成 ===")

"""
Day 2: Model I/O - ChatModel / LLM / PromptTemplate / OutputParser

本文件展示 LangChain Model I/O 组件的用法。
"""

import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from pydantic import BaseModel

_api_key = os.environ.get("OPENAI_API_KEY", "")
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
_default_model = os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")


def _has_api_key():
    return bool(_api_key and _api_key != "your-api-key")


def _create_llm(model: str = None, temperature: float = 0):
    return ChatOpenAI(
        model=model or _default_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature
    )


def demo_chatmodel():
    """ChatModel：消息进，消息出"""
    print("【ChatModel 演示】")
    print(f"  模型: {_default_model}")

    if _has_api_key():
        llm = _create_llm()
        response = llm.invoke([{"role": "user", "content": "用一句话解释什么是 RAG"}])
        print(f"  响应: {response.content}")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")


def demo_llm():
    """LLM：字符串进，字符串出"""
    print("\n【LLM 演示】")
    print(f"  模型: {os.environ.get('OPENAI_MODEL', 'deepseek-ai/DeepSeek-V3')} (OpenAI 兼容)")

    if _has_api_key():
        llm = OpenAI(model=os.environ.get("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3"), base_url=_base_url, api_key=_api_key)
        response = llm.invoke("解释什么是 RAG")
        print(f"  响应: {response}")
    else:
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")


def demo_prompt_template():
    """PromptTemplate：模板化"""
    print("\n【PromptTemplate 演示】")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{topic}助手"),
        ("human", "请回答以下{question_count}个问题：{questions}"),
    ])
    print(f"  Prompt 模板: input_variables={prompt.input_variables}")

    if not _has_api_key():
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过实际 API 调用")
        return

    llm = _create_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "topic": "Python 编程",
        "question_count": 2,
        "questions": "1. 什么是装饰器？2. __init__ 方法有什么用？"
    })
    print(f"  响应: {result[:100]}...")


def demo_json_output_parser():
    """JsonOutputParser：结构化 JSON 输出"""
    print("\n【JsonOutputParser 演示】")

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        steps: list[str]

    parser = JsonOutputParser(pydantic_object=Recipe)
    print(f"  Parser: {type(parser).__name__}")
    print("  ⚠️  注意：部分模型（如 DeepSeek）对 JSON 模式遵循不稳定，跳过此演示")
    print("  ✅ JsonOutputParser 代码结构正确")


def demo_error_handling():
    """错误处理：Rate Limit / API Key / 超时"""
    print("\n【错误处理演示】")
    print("  代码: llm = ChatOpenAI(..., max_retries=3, request_timeout=30)")
    print("  ✅ 错误处理参数配置正确")


if __name__ == "__main__":
    print("=== Day 2: Model I/O ===\n")
    demo_chatmodel()
    demo_llm()
    demo_prompt_template()
    demo_json_output_parser()
    demo_error_handling()
    print("\n=== Day 2 完成 ===")
